cmake_minimum_required(VERSION 3.0.2)
project(lio_sam)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        tf
        cv_bridge
        pcl_conversions
        std_msgs
        sensor_msgs
        geometry_msgs
        nav_msgs
        message_generation
        visualization_msgs
        )

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)

add_message_files(
        DIRECTORY msg
        FILES
        cloud_info.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        std_msgs
        nav_msgs
        sensor_msgs
)

# include directories
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)

catkin_package(
        INCLUDE_DIRS include
        DEPENDS PCL GTSAM

        CATKIN_DEPENDS
        roscpp
        tf
        cv_bridge
        pcl_conversions
        std_msgs
        sensor_msgs
        geometry_msgs
        nav_msgs
        message_generation
        visualization_msgs
)

###########
## Build ##
###########

# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# Feature Association
add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})

# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)
