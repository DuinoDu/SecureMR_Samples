cmake_minimum_required(VERSION 3.22)
project(ufo_origin)

set(SAMPLE_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/cpp/face_tracking_raw.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/helper.cpp
)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)

include(../../base/base.cmake)
