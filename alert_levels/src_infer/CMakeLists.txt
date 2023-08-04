CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(infer_alert_levels)

# Find EDDL
FIND_PACKAGE(EDDL)
IF(NOT EDDL_FOUND)
	MESSAGE(FATAL_ERROR "EDDL not found")
ENDIF(NOT EDDL_FOUND)

# Set the build type to Release
IF (NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type (Debug, Release)" FORCE)
ENDIF (NOT CMAKE_BUILD_TYPE)

# Build infer_alert_levels
ADD_EXECUTABLE(infer_alert_levels infer_alert_levels.cxx)
TARGET_LINK_LIBRARIES(infer_alert_levels PUBLIC EDDL::eddl)

