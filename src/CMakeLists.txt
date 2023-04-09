aux_source_directory( . AUX_SRC_FRONTEND )

# Add slam utility
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
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

# Add feature detector
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector )
if ( NOT TARGET lib_feature_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_feature_detector )
endif()

# Add optical flow tracker
set( OPTICAL_FLOW_TRACKER_PATH ${PROJECT_SOURCE_DIR}/../Optical_Flow_Tracker )
if ( NOT TARGET lib_optical_flow_tracker )
    add_subdirectory( ${OPTICAL_FLOW_TRACKER_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_optical_flow_tracker )
endif()

# Add opencv for debug.
find_package( OpenCV REQUIRED )

add_library( lib_visual_frontend ${AUX_SRC_FRONTEND} )
target_include_directories( lib_visual_frontend PUBLIC . )
target_link_libraries( lib_visual_frontend
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log
    lib_slam_utility_memory

    lib_feature_detector
    lib_optical_flow_tracker

    ${OpenCV_LIBS}
)
