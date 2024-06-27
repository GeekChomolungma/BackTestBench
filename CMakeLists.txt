# CMakeList.txt : CMake project for BurninG, include source and define
# project specific logic here.
#

#cmake -B [build directory] -S [source directory] -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake_minimum_required (VERSION 3.8)

project ("BackTestBench" LANGUAGES CXX)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/strategy/superTrendStrategy ${CMAKE_BINARY_DIR}/library_build)

# 设置运行时库选项
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

# Find the CUDA package
find_package(CUDA)
if(CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    message("CUDA found")
    message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
    message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
    enable_language(CUDA)
    add_definitions(-DHAVE_CUDA)
else()
    message("CUDA not found")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message("PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")

# Find the MONGOCXX package
if(WIN32)
    find_package(bsoncxx CONFIG REQUIRED)
    find_package(mongocxx CONFIG REQUIRED)
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
    "src/strategy/*.h"
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

message("Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
message("Boost_LIBRARIES = ${Boost_LIBRARIES}")

# Add src to path
include_directories(${CMAKE_SOURCE_DIR}/src)

if(WIN32)
    target_link_libraries(BackTestBench PRIVATE $<IF:$<TARGET_EXISTS:mongo::bsoncxx_static>,mongo::bsoncxx_static,mongo::bsoncxx_shared>)
    target_link_libraries(BackTestBench PRIVATE $<IF:$<TARGET_EXISTS:mongo::mongocxx_static>,mongo::mongocxx_static,mongo::mongocxx_shared>)

    if(CUDA_FOUND)
        include_directories("D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common")
        # file(GLOB CUDA_LIBS "D:/cuda12sample/cuda-samples-12.0/cuda-samples-12.0/Common/lib/x64/*.lib")
        # target_link_libraries(BackTestBench PRIVATE ${CUDA_LIBS})
    endif()
else()
    target_include_directories(BackTestBench PRIVATE ${BSONCXX_INCLUDE_DIRS})
    target_include_directories(BackTestBench PRIVATE ${MONGOCXX_INCLUDE_DIRS})
    target_link_libraries(BackTestBench PRIVATE ${BSONCXX_LIBRARIES} ${MONGOCXX_LIBRARIES})

    # set third part depedency
    if(CUDA_FOUND)
        include_directories("/root/dependency/cuda12/cuda-samples-master/Common")
    endif()
endif()

# for config file
configure_file(
    ${CMAKE_SOURCE_DIR}/config.ini
    ${CMAKE_BINARY_DIR}/config.ini
    COPYONLY
)

#target_link_directories(BackTestBench PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(BackTestBench PRIVATE SuperTrendLib)

install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION bin)
