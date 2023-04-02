cmake_minimum_required(VERSION 3.21)

project(gpu-accelerated-sdf-text)

include("./deps/metal-cpp-rendering-boilerplate/dependencies.cmake")

include_metal_cpp_rendering_boilerplate("./deps/metal-cpp-rendering-boilerplate")

add_subdirectory(lib)

set(SRC_LIST
  renderer.cpp
  renderer.hpp
)

set_source_files_properties(${BUNDLE_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${BUNDLE_ICON} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME}
  gpu-accelerated-sdf-text-lib
  common
)
            
target_bundle_msl_libraries(${PROJECT_NAME} gpu-accelerated-sdf-text-lib)

target_enable_arc(${PROJECT_NAME})
