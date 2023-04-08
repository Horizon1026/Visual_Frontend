cmake_minimum_required( VERSION 3.10 )

project( visual_frontend )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )

# Add slam utility
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_slam_utility )
endif()

# Add visual frontend
if ( NOT TARGET lib_visual_frontend )
    add_subdirectory( src ${PROJECT_SOURCE_DIR}/build/lib_visual_frontend )
endif()

# Add test_euroc
add_executable( test_euroc
    test/test_euroc.cpp
)
target_link_libraries( test_euroc
    lib_slam_utility
    lib_visual_frontend
)
