cmake_minimum_required(VERSION 3.8)

project ("SuperTrendLib" LANGUAGES CXX)

# 设置编译输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 设置运行时库选项
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA)
if(CUDA_FOUND)
    message("CUDA found")
    set(CMAKE_CUDA_ARCHITECTURES 75)
    enable_language(CUDA)
    add_definitions(-DHAVE_CUDA)
else()
    message("CUDA not found")
endif()

message("PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")
message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")

# include
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)

# for cuda common tools
if (CUDA_FOUND)
    if(WIN32)
        set(CUDA_COMMON_TOOLS_DIR "D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common")
    else()
        set(CUDA_COMMON_TOOLS_DIR "/root/dependency/cuda12/cuda-samples-master/Common")
    endif()
    message("CUDA_COMMON_TOOLS_DIR = ${CUDA_COMMON_TOOLS_DIR}")
endif()

# create library
add_library(SuperTrendLib STATIC
    src/myStrategy.cpp
)

if (CUDA_FOUND)
    target_include_directories(SuperTrendLib PRIVATE ${CUDA_INCLUDE_DIRS})
    target_include_directories(SuperTrendLib PRIVATE ${CUDA_COMMON_TOOLS_DIR})
    target_sources(SuperTrendLib PRIVATE src/myStrategy.cu)
    # link cuda
    target_link_libraries(SuperTrendLib ${CUDA_LIBRARIES})
endif()