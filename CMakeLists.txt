cmake_minimum_required(VERSION 3.16)
project(sdf_nmpc_ros)

# silence deprecation warning
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(std_srvs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/Latent.msg"
    DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(PROGRAMS
    scripts/build_solver.py
    scripts/vae_node.py
    scripts/ref_gen_node.py
    scripts/sdfnmpc_node.py
    scripts/viz_vae_node.py
    scripts/viz_sdf_2D_node.py
    scripts/viz_sdf_3D_node.py
    DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config rviz
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
