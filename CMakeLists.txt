cmake_minimum_required(VERSION 3.8)
project(sdf_nmpc_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Latent.msg"
  DEPENDENCIES std_msgs
)

# Install Python module files to site-packages
install(
  DIRECTORY sdf_nmpc_ros/
  DESTINATION lib/python3.10/site-packages/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install Python executables with execute permissions
install(
  FILES
    sdf_nmpc_ros/build_solver.py
    sdf_nmpc_ros/sdfnmpc_node.py
    sdf_nmpc_ros/ref_gen_node.py
    sdf_nmpc_ros/vae_node.py
    sdf_nmpc_ros/viz_vae_node.py
  DESTINATION lib/${PROJECT_NAME}
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()