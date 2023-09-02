cmake_minimum_required( VERSION 3.10 )

project( visual_frontend )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

# Add visual frontend
if ( NOT TARGET lib_visual_frontend_mono )
    add_subdirectory( src/frontend_mono ${PROJECT_SOURCE_DIR}/build/lib_visual_frontend_mono )
endif()
if ( NOT TARGET lib_visual_frontend_stereo )
    add_subdirectory( src/frontend_stereo ${PROJECT_SOURCE_DIR}/build/lib_visual_frontend_stereo )
endif()

# Add test_euroc
add_executable( test_euroc
    test/test_euroc.cpp
)
target_link_libraries( test_euroc
    lib_visual_frontend_mono
    lib_visual_frontend_stereo
)
