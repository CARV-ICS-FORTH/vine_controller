/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1]
 *
 * Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *  implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * Links:
 *  ------
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1]
*/
#ifndef OPENCL_ACCELTHREAD
#define OPENCL_ACCELTHREAD

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#define __CL_ENABLE_EXCEPTIONS

#include "timers.h"
#include <CL/cl.hpp>
#include <atomic>
#include <map>
#include <mutex>
#include <pthread.h>

class OpenCLaccelThread;

#include "accelThread.h"

struct CL_file {
	string file;
	bool isBinary;
};

class OpenCLaccelThread : public accelThread {
  public:
	OpenCLaccelThread(vine_pipe_s *v_pipe, AccelConfig &conf);
	~OpenCLaccelThread();
	virtual bool acceleratorInit();	/* Function that initializes a GPU accelerator */
	virtual void acceleratorRelease(); /* Function that resets a GPU accelerator */
	virtual void printOccupancy();
	std::mutex mutexGPUAccess;
	void reset(accelThread *);

  private:
	int pciId;
	vector<CL_file> kernel_files;
	int numberOfPlatforms, numberOfDevices;
	// cl::Device selectedDevice;
	// cl::CommandQueue command_queue;
	bool loadKernels();

	bool initDevice(int id);
	bool getNumberOfDevices();
	bool prepareDevice();
};
#endif
