#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <random>
#include <chrono>
#include <string>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("collision_test_worst_case", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  robot_model_loader::RobotModelLoader robot_model_loader(node, "robot_description");
  const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader.getModel();

  std::vector<int> obstacle_densities = {10, 50, 100, 500};

  std::random_device rd;
  std::mt19937 gen(rd());
  // Poszerzenie strefy losowania i zmniejszenie wymiarów brył
  std::uniform_real_distribution<> dist_x(0.2, 1.2);   
  std::uniform_real_distribution<> dist_y(-1.0, 1.0);  
  std::uniform_real_distribution<> dist_z(0.0, 1.5);   
  std::uniform_real_distribution<> dist_size(0.02, 0.08); 
  std::uniform_int_distribution<> dist_shape(0, 2);
  
  rclcpp::sleep_for(std::chrono::seconds(1));

  for (int density : obstacle_densities) {
    planning_scene::PlanningScene local_scene(kinematic_model);
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

    RCLCPP_INFO(node->get_logger(), "Generowanie %d bezkolizyjnych przeszkod...", density);

    int valid_objects = 0;
    int attempts = 0;

    // Pętla odrzucająca obiekty kolidujące
    while (valid_objects < density && rclcpp::ok()) {
      attempts++;
      moveit_msgs::msg::CollisionObject obj;
      obj.header.frame_id = "base_link";
      obj.id = "obs_" + std::to_string(density) + "_" + std::to_string(valid_objects);
      obj.operation = obj.ADD;

      shape_msgs::msg::SolidPrimitive primitive;
      int shape_type = dist_shape(gen);
      
      if (shape_type == 0) {
        primitive.type = primitive.BOX;
        primitive.dimensions = {dist_size(gen), dist_size(gen), dist_size(gen)};
      } else if (shape_type == 1) {
        primitive.type = primitive.CYLINDER;
        primitive.dimensions = {dist_size(gen), dist_size(gen) / 2.0};
      } else {
        primitive.type = primitive.SPHERE;
        primitive.dimensions = {dist_size(gen) / 2.0};
      }

      geometry_msgs::msg::Pose pose;
      pose.orientation.w = 1.0;
      pose.position.x = dist_x(gen);
      pose.position.y = dist_y(gen);
      pose.position.z = dist_z(gen);

      obj.primitives.push_back(primitive);
      obj.primitive_poses.push_back(pose);

      local_scene.processCollisionObjectMsg(obj);
      
      collision_detection::CollisionRequest req;
      collision_detection::CollisionResult res;
      local_scene.checkCollision(req, res);

      if (res.collision) {
        // Wycofanie obiektu z lokalnej sceny w przypadku kolizji
        obj.operation = obj.REMOVE;
        local_scene.processCollisionObjectMsg(obj);
      } else {
        // Akceptacja bezpiecznej bryły
        collision_objects.push_back(obj);
        valid_objects++;
      }
      
      if (attempts > density * 100) {
        RCLCPP_WARN(node->get_logger(), "Przepełnienie przestrzeni. Przerywam generowanie.");
        break;
      }
    }

    planning_scene_interface.applyCollisionObjects(collision_objects);
    rclcpp::sleep_for(std::chrono::seconds(1));

    // Właściwy pomiar czasu analizy układu
    collision_detection::CollisionRequest req_final;
    collision_detection::CollisionResult res_final;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    local_scene.checkCollision(req_final, res_final);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    RCLCPP_INFO(node->get_logger(), "--> WYNIK (Worst-Case): %d obiektow | Czas FCL: %ld us | Kolizja: %s (Liczba iteracji: %d)", 
                valid_objects, duration.count(), res_final.collision ? "TAK" : "NIE", attempts);

    // Czyszczenie przestrzeni
    std::vector<moveit_msgs::msg::CollisionObject> remove_objects;
    for (auto& obj : collision_objects) {
      obj.operation = obj.REMOVE;
      remove_objects.push_back(obj);
    }
    planning_scene_interface.applyCollisionObjects(remove_objects);
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }

  RCLCPP_INFO(node->get_logger(), "Zakonczono akwizycje danych pomiarowych.");
  rclcpp::shutdown();
  return 0;
}