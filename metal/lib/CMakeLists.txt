cmake_minimum_required(VERSION 3.21)

project(gpu-accelerated-sdf-text-lib)

set(SRC_LIST
  glyph_set.hpp
  glyph_set.mm
  glyph_texture.cpp
  glyph_texture.hpp
  text_renderer.cpp
  text_renderer.hpp
)

set(SRC_LIST_METAL
  sdf_text_types.h
  sdf_text.metal
)

add_library(${PROJECT_NAME} ${SRC_LIST} ${SRC_LIST_METAL})

target_add_msl_library(${PROJECT_NAME} ${SRC_LIST_METAL})

target_enable_arc(${PROJECT_NAME})
