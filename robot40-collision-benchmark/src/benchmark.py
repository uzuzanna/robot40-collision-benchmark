import os
import yaml
import glob
import time

from typing import Annotated, Dict, Any, List
from enum import Enum
import itertools

import typer
from tqdm import tqdm

import numpy as np
import pandas as pd

import open3d as o3d
import trimesh
import igl
import vtk
from vtk.util import numpy_support
from scipy.spatial import Delaunay

from generate_dataset import RandomizationMethod, generate_points
from datetime import datetime
import json
import h5py

class BenchmarkData:
    def __init__(self, 
                 mesh_path: str,
                 points: np.ndarray,
                 voxel_sizes: List[float]):
        self.mesh_path = mesh_path
        self.points = points

        # Load the mesh
        self.mesh_t = trimesh.load(mesh_path, force="mesh")
        self.bounds = self.mesh_t.bounds

        # # Fix the mesh if necessary
        # trimesh.repair.fix_inversion(self.mesh_t)
        # trimesh.repair.fix_winding(self.mesh_t)
        # trimesh.repair.fix_normals(self.mesh_t)
        # if not self.mesh_t.is_watertight:
        #     trimesh.repair.fill_holes(self.mesh_t)

        # # Voxelize
        # self.voxel_grids = {}
        # for voxel_size in voxel_sizes:
        #     voxel_grid = self.mesh_t.voxelized(pitch=voxel_size)
        #     self.voxel_grids[voxel_size] = voxel_grid.fill()

        # Open3D
        self.mesh_o3d = o3d.geometry.TriangleMesh()
        self.mesh_o3d.vertices = o3d.utility.Vector3dVector(self.mesh_t.vertices)
        self.mesh_o3d.triangles = o3d.utility.Vector3iVector(self.mesh_t.faces)
        self.mesh_o3d.remove_duplicated_vertices()
        self.mesh_o3d.compute_vertex_normals()
        self.mesh_o3d.compute_triangle_normals()
        self.o3d_scene = o3d.t.geometry.RaycastingScene()
        self.o3d_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh_o3d))
        self.o3d_tensor_points = o3d.core.Tensor(self.points, dtype=o3d.core.Dtype.Float32)

        # # igl
        # self.igl_vertices = self.mesh_t.vertices
        # self.igl_faces = self.mesh_t.faces

        # # VTK
        # self.vtk_poly = vtk.vtkPolyData()
        # pts = vtk.vtkPoints()
        # pts.SetData(numpy_support.numpy_to_vtk(self.mesh_t.vertices))
        # polys = vtk.vtkCellArray()
        # faces_conn = np.c_[np.full(len(self.mesh_t.faces), 3), self.mesh_t.faces].flatten().astype(np.int64)
        # polys.ImportLegacyFormat(numpy_support.numpy_to_vtkIdTypeArray(faces_conn))
        # self.vtk_poly.SetPoints(pts)
        # self.vtk_poly.SetPolys(polys)

        # self.vtk_points_flat = vtk.vtkPoints()
        # self.vtk_points_flat.SetData(numpy_support.numpy_to_vtk(points))
        # self.vtk_input_poly = vtk.vtkPolyData()
        # self.vtk_input_poly.SetPoints(self.vtk_points_flat)
        
        # self.vtk_select = vtk.vtkSelectEnclosedPoints()
        # self.vtk_select.SetSurfaceData(self.vtk_poly)
        # self.vtk_select.SetInputData(self.vtk_input_poly)

        # # SciPy Convex Hull
        # self.scipy_hull = Delaunay(self.mesh_t.vertices)

def run_trimesh(data: BenchmarkData, **kwargs):
    results = data.mesh_t.contains(data.points)
    return results

def run_trimesh_voxelized(data: BenchmarkData, **kwargs):
    res = kwargs.get('voxelization_res', 0.01)   
    results = data.voxel_grids[res].is_filled(data.points)
    return results

def run_open3d(data: BenchmarkData, **kwargs):
    results = data.o3d_scene.compute_occupancy(data.o3d_tensor_points)
    return results

def run_open3d_sdf(data: BenchmarkData, **kwargs):
    sdf = data.o3d_scene.compute_signed_distance(data.o3d_tensor_points)
    return sdf.numpy() <= 0

def run_igl(data: BenchmarkData, **kwargs):
    wn = igl.fast_winding_number(data.igl_vertices, data.igl_faces, data.points)
    return np.abs(wn) > 0.5

def run_vtk(data: BenchmarkData, **kwargs):
    data.vtk_select.Modified()
    data.vtk_select.Update()
    results = numpy_support.vtk_to_numpy(data.vtk_select.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)
    return results

def run_scipy_convex_hull(data: BenchmarkData, **kwargs):
    simplex = data.scipy_hull.find_simplex(data.points)
    return simplex >= 0

METHOD_MAPPING = {
    "trimesh": run_trimesh,
    "trimesh_voxelized": run_trimesh_voxelized,
    "open3d": run_open3d,
    "open3d_sdf": run_open3d_sdf,
    "igl": run_igl,
    "vtk": run_vtk,
    "scipy_convex_hull": run_scipy_convex_hull
}

def _to_numpy_result(result):
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "numpy"):
        return result.numpy()
    return np.asarray(result)

def _task_to_jsonable(task: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in task.items():
        if isinstance(v, Enum):
            out[k] = v.value
        else:
            out[k] = v
    return out

def main(config_path: Annotated[str, typer.Option("--config_path", "-c", help="Path to the YAML configuration file for benchmark")] = "benchmark_config.yaml"
    ):

    # Load the config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed for reproducibility
    if 'seed' in config:
        np.random.seed(config['seed'])

    # List of mesh files to benchmark
    mesh_files = []
    for mesh_path in config["mesh_path"]:
        mesh_files.extend(glob.glob(mesh_path))
    
    # Validate output file path
    output_file = config["output_file"][0]
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_h5_file = os.path.join(output_dir, f"benchmark_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")

    # Create permutations of parameters for benchmarking
    tasks = []

    for mesh in mesh_files:
        for method_str in config["randomization_method"]:
            method = RandomizationMethod(method_str)

            param_grid = {}
            if method == RandomizationMethod.UNIFORM:
                param_grid = {
                    "num_points": config["num_points"],
                    "sampling_x_dim": config["sampling_x_dim"],
                    "sampling_y_dim": config["sampling_y_dim"],
                    "sampling_z_dim": config["sampling_z_dim"]
                }
            elif method == RandomizationMethod.GAUSSIAN:
                param_grid = {
                    "num_points": config["num_points"],
                    "randomization_std": config["randomization_std"]
                }
            elif method == RandomizationMethod.GRID_WITH_NOISE:
                param_grid = {
                    "grid_sampling_resolution": config["grid_sampling_resolution"],
                    "randomization_std": config["randomization_std"],
                    "sampling_x_dim": config["sampling_x_dim"],
                    "sampling_y_dim": config["sampling_y_dim"],
                    "sampling_z_dim": config["sampling_z_dim"]
                }
            elif method == RandomizationMethod.SURFACE_NOISE:
                param_grid = {
                    "num_points": config["num_points"],
                    "randomization_std": config["randomization_std"]
                }
            
            keys = list(param_grid.keys())
            values = list(param_grid.values())

            # Generate all combinations of parameters
            for combination in itertools.product(*values):
                base_task = dict(zip(keys, combination))
                voxel_resolutions = config.get("voxelization_res", [0.01])

                for voxel_res in voxel_resolutions:
                    task = base_task.copy()
                    task["mesh_path"] = mesh
                    task["randomization_method"] = method
                    task["voxelization_res"] = voxel_res
                    task["seed"] = config["seed"]
                    tasks.append(task)

    typer.secho("*" * 50, fg=typer.colors.GREEN)
    typer.secho(f"Generated {len(tasks)} benchmark tasks. Starting benchmark...", fg=typer.colors.GREEN)
    typer.secho("*" * 50, fg=typer.colors.GREEN)

    # Run benchmarks
    results = []
    curr_mesh_path = None
    trimesh_cache = None
    base_voxel_res = config["voxelization_res"][0]

    methods_to_test = config["methods"]
    num_warmup_runs = config["warmup_runs"][0]
    num_runs = config["num_runs"][0]

    with h5py.File(raw_h5_file, "w") as h5f:
        h5f.attrs["config_path"] = config_path

        for task_idx, task in enumerate(tqdm(tasks, desc="Benchmarking")):

            # Load mesh and prepare data
            if task["mesh_path"] != curr_mesh_path:
                curr_mesh_path = task["mesh_path"]
                trimesh_cache = trimesh.load(curr_mesh_path, force="mesh")

            points = generate_points(task["randomization_method"], task, trimesh_cache)
            data = BenchmarkData(mesh_path=task["mesh_path"], points=points, voxel_sizes=config["voxelization_res"])

            task_group = h5f.require_group(f"tasks/task_{task_idx:05d}")
            if "query_points" not in task_group:
                task_group.create_dataset("query_points", data=points, compression="gzip")
            task_group.attrs["task_config"] = json.dumps(_task_to_jsonable(task))

            for method_name in tqdm(methods_to_test, desc="Methods", leave=False):
                method_func = METHOD_MAPPING[method_name]

                if method_name != "trimesh_voxelized" and task["voxelization_res"] != base_voxel_res:
                    continue

                if not method_func:
                    typer.secho(f"Method {method_name} not recognized. Skipping.", fg=typer.colors.YELLOW)
                    continue

                timings = []

                # Measure runtime
                total_runs = num_warmup_runs + num_runs
                for i in tqdm(range(total_runs), desc="Runs", leave=False):
                    start_time = time.perf_counter_ns()
                    result = method_func(data, **task)
                    end_time = time.perf_counter_ns()

                    if i == num_warmup_runs:
                        method_group = task_group.require_group(f"methods/{method_name}/voxel_res_{task['voxelization_res']}")
                        result_np = _to_numpy_result(result)
                        if "first_run_result" in method_group:
                            del method_group["first_run_result"]
                        method_group.create_dataset("first_run_result", data=result_np, compression="gzip")
                        method_group.attrs["first_run_time_ns"] = int(end_time - start_time)

                    
                    if i >= num_warmup_runs:
                        timings.append(end_time - start_time)

                if timings:
                    avg_ns = np.mean(timings)
                    std_ns = np.std(timings)
                    if len(points) > 0:
                        avg_per_point_ns = avg_ns / len(points)
                    else:
                        avg_per_point_ns = float('inf')
                else:
                    avg_ns = -1
                    std_ns = -1
                    avg_per_point_ns = -1

                # Record results
                rec = task.copy()
                rec["method"] = method_name
                rec["randomization_method"] = task["randomization_method"].value
                rec["avg_time_batch_ns"] = avg_ns
                rec["avg_time_per_point_ns"] = avg_per_point_ns
                rec["std_time_batch_ns"] = std_ns
                rec["all_timings_batch_ns"] = str(timings)
                results.append(rec)

            # Save results to CSV after each task to ensure progress is not lost
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            h5f.flush()

    typer.secho(f"Benchmark completed. Results saved to {output_file}", fg=typer.colors.GREEN)

if __name__ == "__main__":
    typer.run(main)