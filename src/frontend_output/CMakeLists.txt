aux_source_directory( . AUX_SRC_VISUAL_FRONTEND_OUTPUT )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()

add_library( lib_visual_frontend_output
    ${AUX_SRC_VISUAL_FRONTEND_OUTPUT}
)
target_include_directories( lib_visual_frontend_output PUBLIC
    .
)
target_link_libraries( lib_visual_frontend_output
    lib_slam_utility_basic_type
)
