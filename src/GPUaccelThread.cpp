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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <set>
#include <occupancy.cuh>
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

	cudaError_t error;
	int device;
	error = cudaGetDevice(&device);

	if (error != cudaSuccess)
	{
		cerr << "cuda get device FAILED " << endl;
		return -1;
	}
	return device;
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
	/*Find the number of GPUs that exist in the current node*/
	int numberOfGPUS = numberOfCudaDevices();
	static int GPUExistInSystem[128] = {0};

	if (!__sync_bool_compare_and_swap(&GPUExistInSystem[pciId], 0, 1))
	{
		cerr << "Already initialized"<<endl;
		return true;
	}
	if (pciId > numberOfGPUS) {
		cout << "The device with id -" << pciId << "- does not exist!!" << endl;
		cout << "Please set a device (second column in .config) with id smaller than " << numberOfGPUS<< endl;
		cout << "The system wil exit..." << endl;
		return false;
	}

	/*Initilizes a specific GPU*/
	if (initCUDADevice(pciId) == true)
	{
		cout<<"GPU initialization done: "<<pciId<<endl;
	}
	else
	{
		cout << "Failed to set device " << endl;
		return false;
	}
	//int device;
	//cudaGetDevice(&device);
	//cout<<"Device: "<<device<<endl;
	gpuId=getCurrentThreadGpuID();
	resetFlags[gpuId] = false;
	/*Prepare the device */
	if (prepareCUDADevice() == true) {
		cout << "=====================================================" << endl<< endl;
	}
	else
	{
		cout << "Failed to prepare device " << endl;
		return false;
	}

#ifdef SM_OCCUPANCY
	start_event_collection();
#endif

#ifdef SAMPLE_OCCUPANCY
	start_event_collection();
	start_sampling();
#endif

	return true;
}

/*Releases the CPU accelerator*/
void GPUaccelThread::acceleratorRelease() {
	if (resetCUDADevice() == true)
	{
		//cout << "Reset device was successful." << endl;
	}
	else
	{
		cout << "Failed to reset device " << endl;
	}

#ifdef SAMPLE_OCCUPANCY
	stop_sampling();
#endif

}

void GPUaccelThread::printOccupancy() {
#ifdef SM_OCCUPANCY
	get_occupancy();
#endif
}

/**
 * TODO: UGLY HACK TO AVOID API change PLZ FIXME
 */
extern vine_pipe_s *vpipe_s;
/**
 * Transfer Function Implementations
 */
bool Host2GPU(vine_task_msg_s *vine_task, vector<void *> &ioHD)
{
	void *tmpIn;
	bool completed = true;
	cudaError_t errorInputs, errorOutputs;

#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startMalloc, endMalloc, startH2D, endH2D;
	double sumMalloc_In_Out;/*Duration of malloc for all input AND output data*/
	chrono::duration<double, nano> elapsedInputMalloc;
	chrono::duration<double, nano> elapsedMemcpyH2D;
	chrono::duration<double, nano> elapsedMallocOut;
#endif
	/*Map vinedata with cuda data*/
	map<vine_data *, void *> vineData2Cuda;

	utils_breakdown_advance(&(vine_task->breakdown), "cudaMalloc_Inputs");

#ifdef DATA_TRANSFER
	cout<<"Number of inputs: "<<vine_task->in_count<<endl;
	cout<<"Number of outputs: "<<vine_task->out_count<<endl;
#endif

	mutexGPUAccess.lock();

	int mallocIn;
	for (mallocIn = 0; mallocIn < vine_task->in_count; mallocIn++)
	{
		/* Iterate till the number of inputs*/
		if (((((vine_data_s *)vine_task->io[mallocIn].vine_data)->place) & (Both)) == HostOnly)
		{
			ioHD.push_back(vine_data_deref(vine_task->io[mallocIn].vine_data));
			continue;
		}
#ifdef BREAKDOWNS_CONTROLLER
		/*start timer*/
		startMalloc = chrono::system_clock::now();
#endif

		errorInputs = cudaMalloc(&tmpIn, vine_data_size(vine_task->io[mallocIn].vine_data));

		/* Allocate memory to the device for all the inputs*/
		if (errorInputs != cudaSuccess)
		{
			cerr << "cudaMalloc FAILED for input: " << mallocIn << endl;
			/* inform the producer that he has made a mistake*/
			vine_task->state = task_failed;
			completed = false;
			vine_data_mark_ready(vpipe_s ,vine_task->io[mallocIn].vine_data);
			/*In case of failure do not continue to copy*/
			mutexGPUAccess.unlock();
			//usleep(10000);
			throw runtime_error("failed cudaMalloc : Reset");
		}

#ifdef BREAKDOWNS_CONTROLLER
		/*stop timer*/
		endMalloc = chrono::system_clock::now();
		elapsedInputMalloc = endMalloc - startMalloc;
#endif
		/*map between vinedata and cuda alloced data*/
		vineData2Cuda[vine_task->io[mallocIn].vine_data] = tmpIn;
	}

	/*End Malloc input -  Start cudaMemcpy Inputs Host to Device*/
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMemCpy_H2G");

	if (vine_task->state != task_failed)
	{
		int memCpyIn;
		for (memCpyIn = 0; memCpyIn < vine_task->in_count; memCpyIn++)
		{

#ifdef BREAKDOWNS_CONTROLLER
			/*stop timer*/
			startH2D = chrono::system_clock::now();
#endif

			tmpIn=vineData2Cuda[vine_task->io[memCpyIn].vine_data];

			/* Copy inputs to the device */
			if (cudaMemcpy(tmpIn, vine_data_deref(vine_task->io[memCpyIn].vine_data), vine_data_size(vine_task->io[memCpyIn].vine_data), cudaMemcpyHostToDevice) != cudaSuccess)
			{
				cerr << "Cuda Memcpy (Host2Device) FAILED for input: " << memCpyIn << endl;
				vine_task->state = task_failed;
				completed = false;
				vine_data_mark_ready(vpipe_s,vine_task->io[memCpyIn].vine_data);
				mutexGPUAccess.unlock();
				throw runtime_error("failed cudaMemcpy : Reset");
			}

#ifdef DATA_TRANSFER
			cout<<"Size of input " <<memCpyIn<<" is: "<< vine_data_size(vine_task->io[memCpyIn].vine_data)<<endl;
#endif

#ifdef BREAKDOWNS_CONTROLLER
			/*stop timer*/
			endH2D = chrono::system_clock::now();
			elapsedMemcpyH2D = endH2D - startH2D;
#endif
			ioHD.push_back(tmpIn);
		}
		int out;
		void *tmpOut;

		/*End cudaMemCpy Host to Device - Start cudaMalloc for outputs*/
		utils_breakdown_advance(&(vine_task->breakdown), "cudaMalloc_Outputs");

		/*Alocate memory for the outputs */
		for (out = mallocIn; out < vine_task->out_count + mallocIn; out++)
		{
#ifdef BREAKDOWNS_CONTROLLER
			/*start timer*/
			startMalloc = chrono::system_clock::now();
#endif

			if (((vine_data_s *)(vine_task->io[out].vine_data))->flags & VINE_INPUT)
			{
				tmpOut = vineData2Cuda[vine_task->io[out].vine_data];
			}
			else
			{
				errorOutputs = cudaMalloc(&tmpOut, vine_data_size(vine_task->io[out].vine_data));
				if (errorOutputs != cudaSuccess)
				{
					cerr << "cudaMalloc FAILED for output: " << out << endl;
					vine_task->state = task_failed;
					completed = false;
					vine_data_mark_ready(vpipe_s, vine_task->io[out].vine_data);
					mutexGPUAccess.unlock();
					throw runtime_error("failed cudaMalloc : Reset");
				}
				//cudaMemset(tmpOut, 0, vine_data_size(vine_task->io[out].vine_data));
			}

#ifdef BREAKDOWNS_CONTROLLER
			/*stop timer*/
			endMalloc = chrono::system_clock::now();
			elapsedMallocOut = endMalloc - startMalloc;
#endif

			/*End cudaMalloc for outputs - Start Kernel Execution time*/
			ioHD.push_back(tmpOut);

#ifdef BREAKDOWNS_CONTROLLER
			/*malloc duration of all inputs + outputs*/
			sumMalloc_In_Out =  elapsedMallocOut.count() + elapsedInputMalloc.count();
#endif
		}
#ifdef BREAKDOWNS_CONTROLLER
		cout << "---------------Breakdown inside Controller-----------------" << endl;
		cout << "CudaMalloc (inputs + outputs) : " << sumMalloc_In_Out << " nanosec."
			<< endl;
		cout << "CudaMemcpy H2D (inputs) : " << elapsedMemcpyH2D.count() << " nanosec." << endl;
#endif
	}
	/*End cudaMalloc for outputs - Start Kernel Execution time*/
	utils_breakdown_advance(&(vine_task->breakdown), "Kernel_Execution_GPU");

	mutexGPUAccess.unlock();
	return completed;
}

/* Cuda Memcpy from Device to host*/
bool GPU2Host(vine_task_msg_s *vine_task, vector<void *> &ioDH)
{
	int out;
	bool completed = true;

	/*Stop Kernel Execution time - Start cudaMemCpy for outputs Device to Host*/
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMemCpy_G2H");

#ifdef BREAKDOWNS_CONTROLLER
	cudaDeviceSynchronize();
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startD2H, endD2H;
	/*start timer*/
	startD2H = chrono::system_clock::now();
#endif

	mutexGPUAccess.lock();

	for (out = vine_task->in_count; out < vine_task->out_count + vine_task->in_count; out++)
	{

#ifdef DATA_TRANSFER
		cout<<"Size of output " <<out<<" is: "<< vine_data_size(vine_task->io[out].vine_data)<<endl;
#endif

		if (cudaMemcpy(vine_data_deref(vine_task->io[out].vine_data), ioDH[out], vine_data_size(vine_task->io[out].vine_data), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			cerr << "Cuda Memcpy (Device2Host) FAILED for output: " << out << endl;
			vine_task->state = task_failed;
			completed = false;
			vine_data_mark_ready(vpipe_s ,vine_task->io[out].vine_data);
			mutexGPUAccess.unlock();
			throw runtime_error("failed cudaMemcpy : Reset");
		}
		else
		{
			completed = true;
			if (out==vine_task->out_count + vine_task->in_count-1)
			{
				vine_task->state = task_completed;
				/*End cudaMemCpy for outputs Device to Host - Start cudaMemFree*/
				utils_breakdown_advance(&(vine_task->breakdown), "cudaMemFree");
			}
			vine_data_mark_ready(vpipe_s ,vine_task->io[out].vine_data);
		}
	}
#ifdef BREAKDOWNS_CONTROLLER
	/*stop timer*/
	endD2H = chrono::system_clock::now();
	/*duration*/
	chrono::duration<double, nano> elapsed_D2H = endD2H - startD2H;
	cout << "cudaMemcpy D2H (outputs): " << elapsed_D2H.count() << " nanosec."<< endl;
#endif
	mutexGPUAccess.unlock();
	return completed;
}

/* Free Device memory */
bool GPUMemFree(vector<void *> &io)
{
	cudaError_t errorFree ;
#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startFree, endFree;
	/*start timer*/
	startFree = chrono::system_clock::now();
#endif
	mutexGPUAccess.lock();

	bool completed = true;
	set<void *> unique_set(io.begin(), io.end());
	for (set<void *>::iterator itr = unique_set.begin(); itr != unique_set.end(); itr++)
	{

		errorFree = cudaFree(*itr);

		if (errorFree != cudaSuccess)
		{
			cerr << "cudaFree FAILED " << endl;
			completed = false;
			mutexGPUAccess.unlock();
			throw runtime_error("failed cudaFree : Reset");
		}
	}

#ifdef BREAKDOWNS_CONTROLLER
	/*stop timer*/
	endFree = chrono::system_clock::now();
	/*duration*/
	chrono::duration<double, nano> elapsed_Free = endFree - startFree;
	cout << "Free took : " << elapsed_Free.count() << " nanosec" << endl;
	cout<<"------------------ End Breakdown ----------------"<<endl;
#endif

	mutexGPUAccess.unlock();
	return completed;
}

//bool shouldResetGpu (int device)
bool shouldResetGpu ()
{
	int device;
	//	if (device == -1)
	cudaGetDevice(&device);

	//return ( resetFlags[device].exchange(false) );
	return ( resetFlags.at(device).exchange(false) );
}

extern bool resetPolicy ;

cudaError_t my_cudaDeviceSynchronize() __attribute__((used));
cudaError_t my_cudaDeviceSynchronize()
{
	float *d_a;
	int size = sizeof(int);
	cudaError_t error;

	int device =GPUaccelThread::getCurrentThreadGpuID();


	if (!resetPolicy){
		return cudaDeviceSynchronize();
	}

	//cerr << __func__ << endl;
	cudaError_t err = cudaSuccess;
	cudaEvent_t kernelFinished;

	err = cudaEventCreate(&kernelFinished);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda error (cudaEventCreate): %s GPUid: %d.\n", cudaGetErrorString(err), device );
		return err;

	}
	err = cudaEventRecord(kernelFinished);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda error (cudaEventRecord): %s GPUid: %d.\n", cudaGetErrorString(err), device );
		return err;
	}

	while ( (err = cudaEventQuery(kernelFinished)) == cudaErrorNotReady)
	{
	//	usleep(10);

		if ( shouldResetGpu() )
		{
			cerr<<"> Actual Reset!! GPU "<< device <<endl;
			cudaDeviceReset();

			error = cudaMalloc((void**)&d_a,size);
			if (error != cudaSuccess)
			{
				cout <<" First CUDA Malloc Failed at "<< __LINE__ << " .  Error " <<cudaGetErrorString(error)<< " with code  " << error <<endl;
				return error;
			}

			//cudaDeviceSynchronize();
			return err;
		}
	}
	return err;
}

void GPUaccelThread::reset(accelThread * caller)
{
	int jp = 2;
	const char * str[3] = {"BatchJobs","UserJobs","Idle"};
	auto runningTask = getAccelConfig().accelthread->getRunningTask();

	if (runningTask)
	{
		jp = vine_vaccel_get_job_priority((vine_vaccel_s *)(runningTask->accel));
	}

	std::cerr << "NOW" << "(" << getAccelConfig().name << "," << str[jp] << ")"<<" Caller: "<<caller->getAccelConfig().name << std::endl;

	resetFlags.at(pciId) = true;

}

REGISTER_ACCEL_THREAD(GPUaccelThread)
