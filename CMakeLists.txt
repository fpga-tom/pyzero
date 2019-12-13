cmake_minimum_required(VERSION 3.10)
project(muzero)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib64/libtorch-cuda" "/usr/local/lib64/cuda" "abseil-cpp")

find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wl,--no-as-needed")
set(CMAKE_VERBOSE_MAKEFILE ON)
add_subdirectory(abseil-cpp)

add_executable(muzero main.cpp random.h random.cpp)
target_link_libraries(muzero "${TORCH_LIBRARIES}" "${Boost_LIBRARIES}" absl::base absl::synchronization absl::strings)
