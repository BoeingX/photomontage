CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(photomontage)

SET(OpenCV_STATIC OFF)

SET(CMAKE_BUILD_TYPE release)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(photomontage main.cpp photomontage.cpp photomontage.hpp maxflow/graph.cpp)

TARGET_LINK_LIBRARIES(photomontage ${OpenCV_LIBS})
