# CMakeList.txt : CMake project for BurninG, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("BackTestBench" LANGUAGES CXX CUDA)

# TODO: Add tests and install targets if needed.

# Find the CUDA package
# find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the MONGOCXX package
if(WIN32)
    set(BOOST_ROOT "D:/boost/boost_1_83_0")
    #set(bsoncxx_DIR "D:/mongo-driver-cxx/mongo-cxx-driver/lib/cmake/bsoncxx-3.8.0")
    #set(mongocxx_DIR "D:/mongo-driver-cxx/mongo-cxx-driver/lib/cmake/mongocxx-3.8.0")
else()
    find_package(bsoncxx REQUIRED)
    find_package(mongocxx REQUIRED)
endif()


# Add your CUDA source files and other source files
set(SOURCES
    src/config/config.h
    src/main.cpp
)

# Add source to this project's executable.
add_executable(BackTestBench ${SOURCES})

# Add src to path
include_directories(${CMAKE_SOURCE_DIR}/src)

if(WIN32)
  
    # Windows specific settings
    # set third part depedency
    include_directories("D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common")
    include_directories("D:/vcpkg/vcpkg/installed/x64-windows/include/bsoncxx/v_noabi")
    include_directories("D:/vcpkg/vcpkg/installed/x64-windows/include/mongocxx/v_noabi")
    file(GLOB CUDA_LIBS "D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common/lib/x64/*.lib")
    target_link_libraries(BackTestBench ${CUDA_LIBS})
else()
    # Link against the found libraries
    target_link_libraries(BackTestBench "/usr/local/lib/libbsoncxx.so") # ${BSONCXX_LIBRARIES})
    target_link_libraries(BackTestBench "/usr/local/lib/libmongocxx.so") # ${MONGOCXX_LIBRARIES})

    # Add the include directories
    target_include_directories(BackTestBench PRIVATE "/usr/local/include/bsoncxx/v_noabi") # ${BSONCXX_INCLUDE_DIRS})
    target_include_directories(BackTestBench PRIVATE "/usr/local/include/mongocxx/v_noabi") # ${MONGOCXX_INCLUDE_DIRS}) 

    # set third part depedency
    include_directories("/root/dependency/cuda12/cuda-samples-master/Common")
endif()

# for boost environment
find_package(Boost 1.83 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(BackTestBench ${Boost_LIBRARIES})

# for config file
configure_file(
    ${CMAKE_SOURCE_DIR}/config.ini
    ${CMAKE_BINARY_DIR}/config.ini
    COPYONLY
)
