cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME CuRast) 
project(${TARGET_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/common.cmake)

# Paths to TurboJPEG
if (WIN32)
	set(TURBOJPEG_LIBRARY "${CMAKE_SOURCE_DIR}/libs/jpg/turbojpeg-static.lib")
endif()
if (LINUX)
	find_path(TURBOJPEG_INCLUDE_DIR
		NAMES turbojpeg.h
	)
	find_library(TURBOJPEG_LIBRARY
		NAMES turbojpeg
	)
	find_package(TBB REQUIRED)
endif()

# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.[ch]pp
	src/*.cu
	src/ooc_structures/*.[ch]pp
	src/kernels/textureTools.cu
	src/kernels/lines.cu
	#src/kernels/rasterization_utils.cu
)

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})
if (MSVC)
	target_compile_options(${TARGET_NAME} PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
	)
endif()
target_compile_definitions(${TARGET_NAME} PUBLIC
	IMGUI_DEFINE_MATH_OPERATORS
	_HAS_STD_BYTE=0
)
if (WIN32)
	target_compile_definitions(${TARGET_NAME} PUBLIC VK_USE_PLATFORM_WIN32_KHR)
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

target_compile_definitions(${TARGET_NAME}
    PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Include paths
target_include_directories(${TARGET_NAME} PRIVATE
	./src
	./libs
	${TURBOJPEG_INCLUDE_DIR}
)

add_subdirectory(libs/laszip)


if (WIN32)
	target_link_libraries(${TARGET_NAME} PRIVATE 
		${TURBOJPEG_LIBRARY}
		laszip
	)
elseif (LINUX)
	target_link_libraries(${TARGET_NAME} PRIVATE 
		${TURBOJPEG_LIBRARY}
		laszip
		stdc++exp
		TBB::tbb
	)
endif()

# https://stackoverflow.com/questions/2368811/how-to-set-warning-level-in-cmake
target_compile_options(${TARGET_NAME} PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Add other third-party libs
ADD_CUDA(${TARGET_NAME})
ADD_VULKAN(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})

# --------------------------------------------------------
# CUDA configuration
# --------------------------------------------------------
# Choose GPU architectures (adjust for your machine)
# set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)
set_property(TARGET ${TARGET_NAME}
    PROPERTY CUDA_ARCHITECTURES 86)

# Enable separable compilation (for multi-file CUDA builds)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors (optional)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")

# set proper startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})







# Second project
add_executable(PlyToGlb 
	src/tools/PlyToGlb.cpp
	src/unsuck_platform_specific.cpp
)
# target_include_directories(PlyToGlb PRIVATE libs/glm)
target_include_directories(PlyToGlb PRIVATE
	./src
	./libs
)
if (WIN32)
	target_link_libraries(PlyToGlb PRIVATE glfw)
elseif (LINUX)
	target_link_libraries(PlyToGlb PRIVATE glfw TBB::tbb)
endif()