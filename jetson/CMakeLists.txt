# Jetson-side tools. TX2 with JetPack 4.2 (R32.1): cmake 3.10.2, g++ 7.5.
#
# C++14 rather than 17 on purpose: g++ 7.5 only ships std::filesystem behind
# <experimental/filesystem>, and nothing here needs it.
#
# librealsense2 only. No OpenCV link: this box has a header/runtime skew
# (headers 3.3.1, runtime .so 3.2) and the capture path writes raw Y8 anyway.
cmake_minimum_required(VERSION 3.10)
project(doubleeye_jetson CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_options(-Wall -Wextra)

# Prefer the installed CMake package; fall back to pkg-config, which is what
# librealsense 2.22 reliably provides (/usr/local/lib/pkgconfig/realsense2.pc).
find_package(realsense2 QUIET)
if(NOT realsense2_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(REALSENSE2 REQUIRED realsense2)
  add_library(realsense2::realsense2 INTERFACE IMPORTED)
  set_target_properties(realsense2::realsense2 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${REALSENSE2_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${REALSENSE2_LIBRARIES}"
    INTERFACE_LINK_DIRECTORIES "${REALSENSE2_LIBRARY_DIRS}")
  message(STATUS "realsense2 via pkg-config: ${REALSENSE2_VERSION}")
else()
  message(STATUS "realsense2 via CMake package: ${realsense2_VERSION}")
endif()

find_package(Threads REQUIRED)

foreach(tool rs_probe rs_ir_capture)
  add_executable(${tool} src/${tool}.cpp)
  target_include_directories(${tool} PRIVATE src)
  target_link_libraries(${tool} PRIVATE realsense2::realsense2 Threads::Threads)
endforeach()
