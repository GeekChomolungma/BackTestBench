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
    find_package(mongocxx REQUIRED)
    find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

    #set(bsoncxx_DIR "D:/mongo-driver-cxx/mongo-cxx-driver/lib/cmake/bsoncxx-3.8.0")
    #set(mongocxx_DIR "D:/mongo-driver-cxx/mongo-cxx-driver/lib/cmake/mongocxx-3.8.0")
else()
    find_package(bsoncxx REQUIRED)
    find_package(mongocxx REQUIRED)
endif()

# Set Install
set(CMAKE_INSTALL_PREFIX "./App")

# Add your CUDA source files and other source files
set(SOURCES
    src/main.cpp
)

file(GLOB_RECURSE SRC_FILES
    "src/config/*"
    "src/db/*"
    "src/dtos/*"
    "src/platform/*"
    "src/strategy/*"
    "src/indicator/*"
)
list(APPEND SOURCES ${SRC_FILES})

# Add source to this project's executable.
add_executable(BackTestBench ${SOURCES})
install(TARGETS BackTestBench DESTINATION bin)

# Add src to path
include_directories(${CMAKE_SOURCE_DIR}/src)

if(WIN32)
  
    # Windows specific settings
    # set third part depedency
    include_directories("D:/vcpkg/vcpkg/installed/x64-windows/include/bsoncxx/v_noabi")
    include_directories("D:/vcpkg/vcpkg/installed/x64-windows/include/mongocxx/v_noabi")
    target_link_libraries(BackTestBench PRIVATE mongocxx bsoncxx)

    include_directories("D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common")
    file(GLOB CUDA_LIBS "D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common/lib/x64/*.lib")
    target_link_libraries(BackTestBench PRIVATE ${CUDA_LIBS})

    # 设置Anaconda的路径
    set(ANACONDA_PATH "D:/anaconda/release")

    # 手动设置Python解释器的路径
    set(Python3_EXECUTABLE "${ANACONDA_PATH}/python.exe")
    set(Python3_INCLUDE_DIR "${ANACONDA_PATH}/include")
    set(Python3_LIBRARY "${ANACONDA_PATH}/libs/python311.lib") # 根据你的Python版本调整这里

    # 找到Python库
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # 包含Python头文件
    include_directories(${Python3_INCLUDE_DIRS})

    # 如果需要NumPy头文件，手动添加路径
    include_directories("${ANACONDA_PATH}/Lib/site-packages/numpy/core/include")
    target_link_libraries(BackTestBench PRIVATE ${Python3_LIBRARY})

    target_include_directories(BackTestBench PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})
else()
    # Link against the found libraries
    target_link_libraries(BackTestBench PRIVATE "/usr/local/lib/libbsoncxx.so") # ${BSONCXX_LIBRARIES})
    target_link_libraries(BackTestBench PRIVATE "/usr/local/lib/libmongocxx.so") # ${MONGOCXX_LIBRARIES})

    # Add the include directories
    target_include_directories(BackTestBench PRIVATE "/usr/local/include/bsoncxx/v_noabi") # ${BSONCXX_INCLUDE_DIRS})
    target_include_directories(BackTestBench PRIVATE "/usr/local/include/mongocxx/v_noabi") # ${MONGOCXX_INCLUDE_DIRS}) 

    # set third part depedency
    include_directories("/root/dependency/cuda12/cuda-samples-master/Common")
endif()

# for boost environment
find_package(Boost 1.83 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(BackTestBench PRIVATE ${Boost_LIBRARIES})

# for config file
configure_file(
    ${CMAKE_SOURCE_DIR}/config.ini
    ${CMAKE_BINARY_DIR}/config.ini
    COPYONLY
)

install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION bin)
