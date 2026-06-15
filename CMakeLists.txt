cmake_minimum_required(VERSION 3.16)

project(fins_classifier LANGUAGES CXX)

find_package(OpenCV REQUIRED)

fins_add_node(${PROJECT_NAME}
  nodes.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PRIVATE
  MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model"
)
