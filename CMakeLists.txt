cmake_minimum_required(VERSION 3.18)
project(cufhe LANGUAGES CXX CUDA)

# ------------------------------------------------------------------------------
# C++ Compiler Options
# ------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ------------------------------------------------------------------------------
# CUDA Configuration
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set CUDA architectures (modify as needed, currently set for GPU with compute capabilities of 7.0 and 8.0)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80")
endif()

# NVCC (CUDA compiler) options
set(CUDA_NVCC_FLAGS -gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}
    CACHE STRING "NVidia CUDA compiler flags" FORCE)
set(CUDA_NVCC_FLAGS_DEBUG -g -G -O0)
set(CUDA_NVCC_FLAGS_RELEASE -Xptxas -O3 -lineinfo)

# ------------------------------------------------------------------------------
# Options and Compile Definitions
# ------------------------------------------------------------------------------

# Option: Large N (16384)
option(N16384 "Set large N to 16384 (N=16384)" ON)
if(N16384)
    add_compile_definitions(N16384)
endif()

# Option: Scheme selection for N_BUTTERFLY
option(SCHEME "Use Scheme1 (N_BUTTERFLY=1); otherwise Scheme2 (N_BUTTERFLY=8)" "1")
if(SCHEME STREQUAL "1")
    add_compile_definitions(N_BUTTERFLY=1)
elseif(SCHEME STREQUAL "2")
    add_compile_definitions(N_BUTTERFLY=8)
endif()

# Debug Options for cufhe in general
option(DEBUG_NOISELESS "Without noise" OFF)
option(DEBUG_LWE_K0 "0-valued LWE key" OFF)
option(DEBUG_TLWE_K0 "0-valued bootstrapping key" OFF)
option(DEBUG_A0_LWE_ENCRYPT "Debug A0 LWE encryption" OFF)
option(DEBUG_A0_ENCRYPT_ZERO "Debug A0 encryption of zero" OFF)
option(DEBUG_A0_ENCRYPT_EXTERNAL_NOISE "Debug A0 encryption with external noise" OFF)

if(DEBUG_NOISELESS)
    add_compile_definitions(DEBUG_NOISELESS)
endif()
if(DEBUG_LWE_K0)
    add_compile_definitions(DEBUG_LWE_K0)
endif()
if(DEBUG_TLWE_K0)
    add_compile_definitions(DEBUG_TLWE_K0)
endif()
if(DEBUG_A0_LWE_ENCRYPT)
    add_compile_definitions(DEBUG_A0_LWE_ENCRYPT)
endif()
if(DEBUG_A0_ENCRYPT_ZERO)
    add_compile_definitions(DEBUG_A0_ENCRYPT_ZERO)
endif()
if(DEBUG_A0_ENCRYPT_EXTERNAL_NOISE)
    add_compile_definitions(DEBUG_A0_ENCRYPT_EXTERNAL_NOISE)
endif()

# Debug Options for bootstrap
option(DEBUG_ROTATE_A0 "Plaintext Blind Rotate with rotating only b" OFF)
option(DEBUG_ROTATE "Plaintext Blind Rotate without NTT/INTT" OFF)

if(DEBUG_ROTATE_A0)
    add_compile_definitions(DEBUG_ROTATE_A0)
endif()
if(DEBUG_ROTATE)
    add_compile_definitions(DEBUG_ROTATE)
endif()

# Option for Python wrapper
option(PYTHON "Python wrapper" OFF)
if(PYTHON)
    include_directories(${CMAKE_SOURCE_DIR}/cufhe)
endif()

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
set(PYTHON_VERSION "2.7")
set(PYTHON_INCLUDE "/usr/include/python${PYTHON_VERSION}")
set(BOOST_INCLUDE "/usr/include")
include_directories(${PYTHON_INCLUDE} ${BOOST_INCLUDE} ${CMAKE_SOURCE_DIR}/include)

# ------------------------------------------------------------------------------
# Directories for Source, Build, and Tests
# ------------------------------------------------------------------------------
set(DIR_BIN "${CMAKE_SOURCE_DIR}/cufhe/bin")
set(DIR_SRC "${CMAKE_SOURCE_DIR}/cufhe/lib")
set(DIR_TEST "${CMAKE_SOURCE_DIR}/cufhe/test")
set(DIR_PY  "${CMAKE_SOURCE_DIR}/cufhe/python/lib")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIR_BIN})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DIR_BIN})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${DIR_BIN})


# ------------------------------------------------------------------------------
# Helper Function: Set NVCC Flags for a Target
# ------------------------------------------------------------------------------
function(set_cuda_flags TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUDA_NVCC_FLAGS_DEBUG}>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:${CUDA_NVCC_FLAGS_RELEASE}>
    )
endfunction()

# ------------------------------------------------------------------------------
# Build CPU Library (libcufhe_cpu.so)
# ------------------------------------------------------------------------------
file(GLOB CPU_SOURCES
    "${DIR_SRC}/*.cc"
    "${DIR_SRC}/details/*.cc"
)
add_library(cufhe_cpu SHARED ${CPU_SOURCES})
set_target_properties(cufhe_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cufhe_cpu PRIVATE ${CMAKE_SOURCE_DIR}/cufhe)

# ------------------------------------------------------------------------------
# Build GPU Library (libcufhe_gpu.so)
# ------------------------------------------------------------------------------
file(GLOB GPU_CU_SOURCES
    "${DIR_SRC}/*.cu"
    "${DIR_SRC}/details/*.cu"
    "${DIR_SRC}/ntt_gpu/*.cu"
)
set(GPU_CPU_SOURCES
    "${DIR_SRC}/cufhe.cc"
    "${DIR_SRC}/cufhe_io.cc"
)
set(GPU_SOURCES 
    ${GPU_CU_SOURCES} 
    ${GPU_CPU_SOURCES}
)
add_library(cufhe_gpu SHARED ${GPU_SOURCES})
target_include_directories(cufhe_gpu PRIVATE ${CMAKE_SOURCE_DIR}/cufhe)
# Apply NVCC flags to the GPU library target
set_cuda_flags(cufhe_gpu)

# ------------------------------------------------------------------------------
# Build Test Executables
# ------------------------------------------------------------------------------

# Test: CPU API
add_executable(test_api_cpu ${DIR_TEST}/test_api_cpu.cc)
target_include_directories(test_api_cpu PRIVATE ${CMAKE_SOURCE_DIR}/cufhe)
target_link_libraries(test_api_cpu PRIVATE cufhe_cpu)

# Test: GPU API
add_executable(test_api_gpu ${DIR_TEST}/test_api_gpu.cu)
target_include_directories(test_api_gpu PRIVATE ${CMAKE_SOURCE_DIR}/cufhe)
target_link_libraries(test_api_gpu PRIVATE cufhe_gpu)
set_cuda_flags(test_api_gpu)

# Test: NTT GPU
add_executable(test_ntt_gpu ${DIR_TEST}/test_ntt_gpu.cu)
target_include_directories(test_ntt_gpu PRIVATE ${CMAKE_SOURCE_DIR}/cufhe)
target_link_libraries(test_ntt_gpu PRIVATE cufhe_gpu)
set_cuda_flags(test_ntt_gpu)
