cmake_minimum_required(VERSION 3.20)
project(CPP_Image VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inclure Eigen
include_directories("C:/libs/eigen-3.4.0")

add_executable(MonProjet src/test.cpp)
