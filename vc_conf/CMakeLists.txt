cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET)

if(CUDA_FOUND)
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
	cuda_add_executable(
		vc_conf GpuDetect.cu vc_conf.cpp Accel.cpp)
else(CUDA_FOUND)
	add_executable(
		vc_conf NoGpuDetect.cpp vc_conf.cpp Accel.cpp)
endif(CUDA_FOUND)
