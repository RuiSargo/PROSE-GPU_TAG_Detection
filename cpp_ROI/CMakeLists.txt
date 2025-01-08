cmake_minimum_required(VERSION 3.10)
project(ArUcoDetector)

# Caminho do OpenCV
set(OpenCV_DIR "C:/opencv2/opencv/build")

# Encontrar o OpenCV
find_package(OpenCV REQUIRED)

# Incluir diretórios
include_directories(${OpenCV_INCLUDE_DIRS})

# Adicionar o executável
add_executable(ArUcoDetector src/main.cpp)

# Ligar bibliotecas do OpenCV
target_link_libraries(ArUcoDetector ${OpenCV_LIBS})
