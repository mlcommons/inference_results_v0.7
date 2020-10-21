cmake_minimum_required(VERSION 2.8)

project(ImageNet)

include_directories(/usr/local/include)

find_package( OpenCV REQUIRED )

set(HEADER_FILES
        ImageNetLabels.h
        ImageNetUtils.h
        )

set(SOURCE_FILES
        ImageNetLabels.cpp
        ImageNetUtils.cpp
        )

add_library(imagenet STATIC ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(imagenet ${OpenCV_LIBS})