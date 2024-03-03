# CMakeList.txt : CMake project for BurninG, include source and define
# project specific logic here.
#

#cmake -B [build directory] -S [source directory] -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake

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
    #set(Boost_NO_SYSTEM_PATHS ON)
    #set(Boost_NO_BOOST_CMAKE ON)  # 禁用 Boost 的 CMake 配置，以便使用自定义路径
    set(BOOST_ROOT "D:/boost/boost_1_83_0")
    set(BOOST_LIBRARYDIR "D:/boost/boost_1_83_0/stage/lib") # ${BOOST_ROOT}/stage/lib
    
    #find_package(mongocxx REQUIRED)
    find_package(libmongocxx REQUIRED)
    find_package(libbsoncxx REQUIRED)

    message("LIBMONGOCXX_INCLUDE_DIRS = ${LIBMONGOCXX_INCLUDE_DIRS}")
    message("LIBMONGOCXX_LIBRARIES = ${LIBMONGOCXX_LIBRARIES}")
    message("LIBBSONCXX_INCLUDE_DIRS = ${LIBBSONCXX_INCLUDE_DIRS}")
    message("LIBBSONCXX_LIBRARIES = ${LIBBSONCXX_LIBRARIES}")

    # matplot
    find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(MONGOCXX REQUIRED libmongocxx)
    pkg_check_modules(BSONCXX REQUIRED libbsoncxx)
    include_directories(${MONGOCXX_INCLUDE_DIRS} ${BSONCXX_INCLUDE_DIRS})
    link_directories(${MONGOCXX_LIBRARY_DIRS} ${BSONCXX_LIBRARY_DIRS})
    add_definitions(${MONGOCXX_CFLAGS_OTHER} ${BSONCXX_CFLAGS_OTHER})

    message("BSONCXX_INCLUDE_DIRS = ${BSONCXX_INCLUDE_DIRS}")
    message("MONGOCXX_INCLUDE_DIRS = ${MONGOCXX_INCLUDE_DIRS}")
    message("BSONCXX_LIBRARIES = ${BSONCXX_LIBRARIES}")
    message("MONGOCXX_LIBRARIES = ${MONGOCXX_LIBRARIES}")
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
    "src/threadPool/*"
    "src/settlement/*"
)
list(APPEND SOURCES ${SRC_FILES})

# Add source to this project's executable.
add_executable(BackTestBench ${SOURCES})
install(TARGETS BackTestBench DESTINATION bin)

# for boost environment
find_package(Boost REQUIRED COMPONENTS thread)
# target_link_libraries(BackTestBench PRIVATE Boost::thread)
# above command equals including libs and headers both
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(BackTestBench PRIVATE ${Boost_LIBRARIES})

# # 使用 target_include_directories 而不是 include_directories
# target_include_directories(BackTestBench PRIVATE ${Boost_INCLUDE_DIRS})
# # 直接链接到 Boost imported targets
# target_link_libraries(BackTestBench PRIVATE Boost::thread)

# Add src to path
include_directories(${CMAKE_SOURCE_DIR}/src)

if(WIN32)
  
    # Windows specific settings
    # set third part depedency
    # include_directories("D:/vcpkg/vcpkg/installed/x64-windows/include/bsoncxx/v_noabi")
    # include_directories("D:/vcpkg/vcpkg/installed/x64-windows/include/mongocxx/v_noabi")
    # target_link_libraries(BackTestBench PRIVATE mongocxx bsoncxx)
    target_include_directories(BackTestBench PRIVATE ${LIBBSONCXX_INCLUDE_DIRS})
    target_include_directories(BackTestBench PRIVATE ${LIBMONGOCXX_INCLUDE_DIRS})
    target_link_libraries(BackTestBench PRIVATE ${LIBBSONCXX_LIBRARIES} ${LIBMONGOCXX_LIBRARIES})

    include_directories("D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common")
    file(GLOB CUDA_LIBS "D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common/lib/x64/*.lib")
    target_link_libraries(BackTestBench PRIVATE ${CUDA_LIBS})

    # # 设置Anaconda的路径
    # set(ANACONDA_PATH "D:/anaconda/release")

    # # 手动设置Python解释器的路径
    # set(Python3_EXECUTABLE "${ANACONDA_PATH}/python.exe")
    # set(Python3_INCLUDE_DIR "${ANACONDA_PATH}/include")
    # set(Python3_LIBRARY "${ANACONDA_PATH}/libs/python311.lib") # 根据你的Python版本调整这里

    # # 找到Python库
    # find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # # 包含Python头文件
    # include_directories(${Python3_INCLUDE_DIRS})

    # # 如果需要NumPy头文件，手动添加路径
    # include_directories("${ANACONDA_PATH}/Lib/site-packages/numpy/core/include")
    # target_link_libraries(BackTestBench PRIVATE ${Python3_LIBRARY})

    # target_include_directories(BackTestBench PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})
else()
    # # Link against the found libraries
    # target_link_libraries(BackTestBench PRIVATE "/usr/local/lib/libbsoncxx.so") # ${BSONCXX_LIBRARIES})
    # target_link_libraries(BackTestBench PRIVATE "/usr/local/lib/libmongocxx.so") # ${MONGOCXX_LIBRARIES})

    # # Add the include directories
    # target_include_directories(BackTestBench PRIVATE "/usr/local/include/bsoncxx/v_noabi") # ${BSONCXX_INCLUDE_DIRS})
    # target_include_directories(BackTestBench PRIVATE "/usr/local/include/mongocxx/v_noabi") # ${MONGOCXX_INCLUDE_DIRS}) 

    target_include_directories(BackTestBench PRIVATE ${BSONCXX_INCLUDE_DIRS})
    target_include_directories(BackTestBench PRIVATE ${MONGOCXX_INCLUDE_DIRS})
    target_link_libraries(BackTestBench PRIVATE ${BSONCXX_LIBRARIES} ${MONGOCXX_LIBRARIES})

    # set third part depedency
    include_directories("/root/dependency/cuda12/cuda-samples-master/Common")
endif()

# for config file
configure_file(
    ${CMAKE_SOURCE_DIR}/config.ini
    ${CMAKE_BINARY_DIR}/config.ini
    COPYONLY
)

install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION bin)
