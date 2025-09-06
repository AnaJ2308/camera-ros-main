cmake_minimum_required(VERSION 3.16)
project(hfield_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# ---- hello_node (no MuJoCo) ----
add_executable(hello_node src/hello_node.cpp)
ament_target_dependencies(hello_node rclcpp)
install(TARGETS hello_node DESTINATION lib/${PROJECT_NAME})

# ---- hfield_node (MuJoCo) ----
if(DEFINED MUJOCO_INCLUDE_DIR AND DEFINED MUJOCO_LIB)
  message(STATUS "Using MuJoCo include: ${MUJOCO_INCLUDE_DIR}")
  message(STATUS "Using MuJoCo lib    : ${MUJOCO_LIB}")
  include_directories(${MUJOCO_INCLUDE_DIR})

  add_executable(hfield_node src/hfield_node.cpp)
  ament_target_dependencies(hfield_node rclcpp sensor_msgs)
  target_link_libraries(hfield_node ${MUJOCO_LIB})

  # rpath so runtime can find the .so without LD_LIBRARY_PATH
  get_filename_component(_MJ_LIBDIR "${MUJOCO_LIB}" DIRECTORY)
  target_link_options(hfield_node PRIVATE "-Wl,-rpath,${_MJ_LIBDIR}")

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLFW3 REQUIRED glfw3)
  target_include_directories(hfield_node PRIVATE ${GLFW3_INCLUDE_DIRS})
  target_link_libraries(hfield_node ${MUJOCO_LIB} ${GLFW3_LIBRARIES})


  install(TARGETS hfield_node DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING "MUJOCO_INCLUDE_DIR / MUJOCO_LIB not set; skipping hfield_node build")
endif()

ament_package()
