aux_source_directory( . AUX_SRC_FRONTEND_STEREO )
aux_source_directory( .. AUX_SRC_FRONTEND_BASIC )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if ( NOT TARGET lib_slam_utility_memory )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/memory ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_memory )
endif()

# Add all relative components of slam utility data type.
if ( NOT TARGET lib_image )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/image ${PROJECT_SOURCE_DIR}/build/lib_image )
endif()
if ( NOT TARGET lib_image_pyramid )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/image_pyramid ${PROJECT_SOURCE_DIR}/build/lib_image_pyramid )
endif()

# Add visualizor.
set( VISUALIZOR_PATH ${PROJECT_SOURCE_DIR}/../Visualizor2D )
if ( NOT TARGET lib_2d_visualizor )
    add_subdirectory( ${VISUALIZOR_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_2d_visualizor )
endif()

# Add feature detector and feature descriptor.
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector )
if ( NOT TARGET lib_feature_point_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_point_detector )
endif()

# Add optical flow tracker and descriptor matcher.
set( OPTICAL_FLOW_TRACKER_PATH ${PROJECT_SOURCE_DIR}/../Feature_Tracker )
if ( NOT TARGET lib_optical_flow_tracker )
    add_subdirectory( ${OPTICAL_FLOW_TRACKER_PATH}/src/optical_flow_tracker ${PROJECT_SOURCE_DIR}/build/lib_optical_flow_tracker )
endif()
if ( NOT TARGET lib_descriptor_matcher )
    add_subdirectory( ${OPTICAL_FLOW_TRACKER_PATH}/src/descriptor_matcher ${PROJECT_SOURCE_DIR}/build/lib_descriptor_matcher )
endif()

# Add all relative components of vision geometry.
set( VISION_GEOMETRY_PATH ${PROJECT_SOURCE_DIR}/../Vision_Geometry )
if ( NOT TARGET lib_vision_geometry_epipolar )
    add_subdirectory( ${VISION_GEOMETRY_PATH}/src/epipolar ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_epipolar )
endif()
if ( NOT TARGET lib_vision_geometry_pnp )
    add_subdirectory( ${VISION_GEOMETRY_PATH}/src/perspective_n_point ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_pnp )
endif()
if ( NOT TARGET lib_vision_geometry_point_triangulator )
    add_subdirectory( ${VISION_GEOMETRY_PATH}/src/point_triangulator ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_point_triangulator )
endif()

# Add camera model.
set( SENSOR_CAMERA_MODEL_PATH ${PROJECT_SOURCE_DIR}/../Sensor_Model )
if ( NOT TARGET lib_camera_model )
    add_subdirectory( ${SENSOR_CAMERA_MODEL_PATH}/src/camera ${PROJECT_SOURCE_DIR}/build/lib_camera_model )
endif()

# Add image processor.
set( IMAGE_PROCESSOR_PATH ${PROJECT_SOURCE_DIR}/../Image_Processor )
if ( NOT TARGET lib_image_processor )
    add_subdirectory( ${IMAGE_PROCESSOR_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_image_processor )
endif()

# Add binary data logger.
set( BINARY_DATA_LOG_PATH ${PROJECT_SOURCE_DIR}/../Binary_Data_Log )
if ( NOT TARGET lib_binary_data_log )
    add_subdirectory( ${BINARY_DATA_LOG_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_binary_data_log )
endif()

add_library( lib_visual_frontend_stereo
    ${AUX_SRC_FRONTEND_STEREO}
    ${AUX_SRC_FRONTEND_BASIC}
)
target_include_directories( lib_visual_frontend_stereo PUBLIC
    .
    ..
)
target_link_libraries( lib_visual_frontend_stereo
    lib_slam_utility_basic_type
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log
    lib_slam_utility_memory

    lib_image
    lib_image_pyramid

    lib_2d_visualizor

    lib_feature_point_detector

    lib_optical_flow_tracker
    lib_descriptor_matcher

    lib_vision_geometry_epipolar
    lib_vision_geometry_pnp
    lib_vision_geometry_point_triangulator

    lib_camera_model

    lib_image_processor

    lib_binary_data_log
)
