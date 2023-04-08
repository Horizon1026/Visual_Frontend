aux_source_directory( . AUX_SRC_FRONTEND )

if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()

if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()

if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()

if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

if ( NOT TARGET lib_slam_utility_memory )
    add_subdirectory( memory ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_memory )
endif()

add_library( lib_visual_frontend ${AUX_SRC_FRONTEND} )
target_include_directories( lib_visual_frontend PUBLIC . )
target_link_libraries( lib_visual_frontend
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log
    lib_slam_utility_memory
)
