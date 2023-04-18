aux_source_directory( . AUX_SRC_FRONTEND )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if ( NOT TARGET lib_slam_utility_memory )
    add_subdirectory( ${SLAM_UTILITY_PATH}/memory ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_memory )
endif()

# Add feature detector.
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector )
if ( NOT TARGET lib_feature_point_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_point_detector )
endif()

# Add optical flow tracker.
set( OPTICAL_FLOW_TRACKER_PATH ${PROJECT_SOURCE_DIR}/../Feature_Tracker )
if ( NOT TARGET lib_optical_flow_tracker )
    add_subdirectory( ${OPTICAL_FLOW_TRACKER_PATH}/src/optical_flow_tracker ${PROJECT_SOURCE_DIR}/build/lib_optical_flow_tracker )
endif()

# Add vision geometry.
set( VISION_GEOMETRY_PATH ${PROJECT_SOURCE_DIR}/../Vision_Geometry )
if ( NOT TARGET lib_vision_geometry )
    add_subdirectory( ${VISION_GEOMETRY_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry )
endif()

# Add camera model.
set( SENSOR_CAMERA_MODEL_PATH ${PROJECT_SOURCE_DIR}/../Sensor_Model )
if ( NOT TARGET lib_camera_model )
    add_subdirectory( ${SENSOR_CAMERA_MODEL_PATH}/src/camera ${PROJECT_SOURCE_DIR}/build/lib_camera_model )
endif()

add_library( lib_visual_frontend ${AUX_SRC_FRONTEND} )
target_include_directories( lib_visual_frontend PUBLIC . )
target_link_libraries( lib_visual_frontend
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log
    lib_slam_utility_memory

    lib_feature_point_detector
    lib_optical_flow_tracker
    lib_vision_geometry
    lib_camera_model
)
