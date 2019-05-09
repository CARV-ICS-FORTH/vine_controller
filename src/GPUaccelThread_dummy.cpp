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
#include <iostream>
#include "prepareGPU.cuh"
#include "GPUaccelThread.h"
#include "definesEnable.h"
#include "utils/timer.h"
#include <set>
#include <mutex>
#include <exception>

using namespace std;
std::mutex mutexGPUAccess;

map <int, atomic<bool>> resetFlags;

GPUaccelThread::GPUaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf) : accelThread(v_pipe, conf)
{
	this->pciId=atoi(conf.init_params.c_str());
	//	resetFlags[this->pciId] = false;
}

GPUaccelThread::~GPUaccelThread() {}

int GPUaccelThread::getCurrentThreadGpuID()
{

	return 0;
}

int GPUaccelThread::getObjGpuId()
{
	return gpuId;   
}

bool GPUaccelThread::getGpuResetState(int device)
{
	return resetFlags[device];	
}

/*initializes the GPU accelerator*/
bool GPUaccelThread::acceleratorInit()
{
	return false;
}

/*Releases the CPU accelerator*/
void GPUaccelThread::acceleratorRelease() {
	
}

void GPUaccelThread::printOccupancy() {
}

//bool shouldResetGpu (int device)
bool shouldResetGpu ()
{
	return false;
}

extern bool resetPolicy ;

void GPUaccelThread::reset(accelThread * caller)
{
}

//REGISTER_ACCEL_THREAD(GPUaccelThread)
