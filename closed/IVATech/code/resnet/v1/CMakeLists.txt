cmake_minimum_required(VERSION 3.10)

set(LOADGEN_PATH $ENV{LOADGEN_PATH})
if(NOT LOADGEN_PATH)
    message( FATAL_ERROR "LOADGEN_PATH is not defined" )
endif()

project(mlperf_resnet50)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -std=c++1z")

add_subdirectory(imagenet)

find_package (TBB HINTS /opt/intel/tbb REQUIRED)
find_package(spdlog HINTS /opt/spdlog REQUIRED)

find_package(clipp HINTS /opt/clipp REQUIRED)
if(NOT ${clipp_FOUND})
    message(FATAL_ERROR "clipp library not found")
endif()

include_directories(${LOADGEN_PATH} /opt/clipp/include ${TBB_includes} /opt/spdlog/include)
link_directories(${LOADGEN_PATH})

add_executable(iva_resnet50 main.cpp IvaTPUSystemUnderTest.cpp IvaTPUQuerySampleLibrary.cpp util.h util.cpp)
target_link_libraries(iva_resnet50 PUBLIC mlperf_loadgen pthread glog imagenet tpu stdc++fs ${TBB_IMPORTED_TARGETS} PRIVATE spdlog::spdlog)

find_package(Catch2 REQUIRED)
add_executable(test_mlperf tests/test_main.cpp tests/test_inference.cpp util.h util.cpp IvaTPUQuerySampleLibrary.cpp IvaTPUSystemUnderTest.cpp)
target_link_libraries(test_mlperf Catch2::Catch2 tpu stdc++fs pthread imagenet)
target_include_directories(test_mlperf PRIVATE ${PROJECT_SOURCE_DIR})
include(CTest)
include(Catch)
catch_discover_tests(test_mlperf)