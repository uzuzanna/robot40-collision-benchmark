# Robot 4.0 Collision Detection Benchmark

Projekt zrealizowany w ramach przedmiotu: Metody i Algorytmy Planowania Ruchu.
Celem projektu było porównanie wydajności dwóch metod detekcji kolizji dla robota mobilno-manipulacyjnego Robot 4.0.

## Struktura projektu
- `/src`: Kody źródłowe (C++ dla MoveIt/FCL, Python dla Open3D/Trimesh)
- `/config`: Pliki konfiguracyjne YAML dla benchmarku
- `/results`: Zestawienie wyników pomiarów w formacie CSV

## Główne wnioski
- Metoda FCL (C++) oferuje najniższe czasy detekcji (sub-milisekundowe).
- Metoda Open3D SDF jest optymalnym rozwiązaniem dla danych z chmury punktów w czasie rzeczywistym.
- Złożone siatki CAD bez decymacji powodują przepełnienie pamięci RAM (OOM Killer).
