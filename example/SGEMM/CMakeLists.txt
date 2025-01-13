#
# Copyright 2025 Philippe Felix Haupt, Eric-Ramon Kreyer, Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME SGEMM)

project(${_TARGET_NAME} LANGUAGES CXX)



#-------------------------------------------------------------------------------
# Find alpaka.

if (NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if (alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else ()
        find_package(alpaka REQUIRED)
    endif ()
endif ()

#-------------------------------------------------------------------------------
# Add executable.

file(GLOB_RECURSE sgemmSource "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach (gmmExampleFile ${sgemmSource})
    get_filename_component(gmmFileName ${gmmExampleFile} NAME)
    string(REPLACE ".cpp" "" gmmName ${gmmFileName})
    alpaka_add_executable(${gmmName} ${gmmExampleFile})
    target_link_libraries(${gmmName} PUBLIC alpaka::alpaka)
    set_target_properties(${gmmName} PROPERTIES FOLDER sgemm)
    target_compile_features(${gmmName} PRIVATE cxx_std_20)

    add_test(NAME ${gmmName} COMMAND ${gmmName})
endforeach ()