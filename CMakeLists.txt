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
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)



# ---------- multicam_hfield (MuJoCo + GLFW + TF2) ----------
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)
find_package(OpenGL REQUIRED)

# MuJoCo discovery: honor passed-in vars, else try common paths
if(NOT DEFINED MUJOCO_INCLUDE_DIR)
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    HINTS $ENV{MUJOCO_INCLUDE_DIR}
          /usr/include /usr/local/include /opt/homebrew/include)
endif()

if(NOT DEFINED MUJOCO_LIB)
  find_library(MUJOCO_LIB mujoco
    HINTS $ENV{MUJOCO_LIB_DIR}
          /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
endif()

# ---------- multicam_no_tf (MuJoCo + GLFW, NO TF2) ----------
if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIB)
  add_executable(multicam_no_tf src/multicam_no_tf.cpp)

  target_include_directories(multicam_no_tf PRIVATE
    ${MUJOCO_INCLUDE_DIR}
    ${GLFW3_INCLUDE_DIRS}
  )

  # Note: no tf2 dependencies here
  ament_target_dependencies(multicam_no_tf
    rclcpp
    sensor_msgs
  )

  target_link_libraries(multicam_no_tf
    ${MUJOCO_LIB}
    ${GLFW3_LIBRARIES}
    OpenGL::GL
    dl
    m
    pthread
  )

  # Ensure libmujoco.so is found at runtime (same as other target)
  target_link_options(multicam_no_tf PRIVATE "-Wl,-rpath,${_MJ_LIBDIR}")

  install(TARGETS multicam_no_tf DESTINATION lib/${PROJECT_NAME})
endif()

# ---------- onecam_hfield_tf (MuJoCo + GLFW + TF2) ----------
if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIB)
  add_executable(onecam_hfield_tf src/onecam_hfield_tf.cpp)

  target_include_directories(onecam_hfield_tf PRIVATE
    ${MUJOCO_INCLUDE_DIR}
    ${GLFW3_INCLUDE_DIRS}
  )

  ament_target_dependencies(onecam_hfield_tf
    rclcpp
    sensor_msgs
    tf2_ros
    tf2_sensor_msgs
  )

  target_link_libraries(onecam_hfield_tf
    ${MUJOCO_LIB}
    ${GLFW3_LIBRARIES}
    OpenGL::GL
    dl
    m
    pthread
  )

  # Reuse rpath from your existing target
  get_filename_component(_MJ_LIBDIR "${MUJOCO_LIB}" DIRECTORY)
  target_link_options(onecam_hfield_tf PRIVATE "-Wl,-rpath,${_MJ_LIBDIR}")

  install(TARGETS onecam_hfield_tf DESTINATION lib/${PROJECT_NAME})
endif()

# ---------- onecam_mytf (MuJoCo + GLFW, NO TF lookup; hardcoded R|t) ----------
if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIB)
  add_executable(onecam_mytf src/onecam_mytf.cpp)

  target_include_directories(onecam_mytf PRIVATE
    ${MUJOCO_INCLUDE_DIR}
    ${GLFW3_INCLUDE_DIRS}
  )

  ament_target_dependencies(onecam_mytf
    rclcpp
    sensor_msgs
  )

  target_link_libraries(onecam_mytf
    ${MUJOCO_LIB}
    ${GLFW3_LIBRARIES}
    OpenGL::GL
    dl m pthread
  )

  if(_MJ_LIBDIR)
    target_link_options(onecam_mytf PRIVATE "-Wl,-rpath,${_MJ_LIBDIR}")
  endif()

  install(TARGETS onecam_mytf DESTINATION lib/${PROJECT_NAME})
endif()



if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIB)
  message(STATUS "Using MuJoCo include: ${MUJOCO_INCLUDE_DIR}")
  message(STATUS "Using MuJoCo lib    : ${MUJOCO_LIB}")

  add_executable(multicam_hfield src/multicam_hfield.cpp)

  target_include_directories(multicam_hfield PRIVATE
    ${MUJOCO_INCLUDE_DIR}
    ${GLFW3_INCLUDE_DIRS}
  )

  ament_target_dependencies(multicam_hfield
    rclcpp
    sensor_msgs
    tf2_ros
    tf2_sensor_msgs
  )

  target_link_libraries(multicam_hfield
    ${MUJOCO_LIB}
    ${GLFW3_LIBRARIES}
    OpenGL::GL
    dl
    m
    pthread
  )

  # Embed rpath so libmujoco.so is found at runtime
  get_filename_component(_MJ_LIBDIR "${MUJOCO_LIB}" DIRECTORY)
  target_link_options(multicam_hfield PRIVATE "-Wl,-rpath,${_MJ_LIBDIR}")

  install(TARGETS multicam_hfield DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING "MuJoCo not found. Set MUJOCO_INCLUDE_DIR and MUJOCO_LIB (or install system-wide). Skipping multicam_hfield.")
endif()



ament_package()
