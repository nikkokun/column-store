cmake_minimum_required(VERSION 3.14)
project(column_store)

set (Boost_USE_STATIC_LIBS OFF)
FIND_PACKAGE(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 17)

add_executable(column_store main.cpp)
target_link_libraries(column_store ${Boost_INCLUDE_DIRS})