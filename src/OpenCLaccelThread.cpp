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
// #define CL_HPP_ENABLE_EXCEPTIONS
// #define CL_HPP_TARGET_OPENCL_VERSION 120
// #define CL_HPP_MINIMUM_OPENCL_VERSION 110
// #include <CL/cl2.hpp>


#include "OpenCLaccelThread.h"
#include "Utilities.h"
#include "definesEnable.h"
#include "err_code.h"
#include "prepareGPU.cuh"
#include "utils/timer.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>

using namespace std;
mutex mutexOpenCLAccess;

map<int, atomic<bool>> OpenCLresetFlags;
map<string, cl::Kernel> kernels;

cl::Platform defaultPlatform;
cl::Device defaultDevice;
cl::Context defaultContext;
cl::CommandQueue defaultCommandQueue;


inline std::string loadFile(std::string input) {
	std::ifstream stream(input.c_str(), ios::in | ios::binary);
	if (!stream.is_open()) {
		std::cout << "Cannot open file: " << input << std::endl;
		exit(1);
	}

	return std::string(
		std::istreambuf_iterator<char>(stream),
		(std::istreambuf_iterator<char>()));
}

vector<string> decodeFilesArgs(string filesArgs) {
	vector<string> vect;

	stringstream ss(filesArgs);
	while (ss.good()) {
		string file;
		getline(ss, file, ',');
		file = trim(file);
		vect.push_back(file);
	}

	return vect;
}

vector<CL_file> loadFiles(vector<std::string> files_funcs) {
	bool isBinary;
	vector<CL_file> kernel_files;
	for (std::vector<std::string>::iterator it = files_funcs.begin(); it != files_funcs.end(); it++) {
		std::string f = loadFile(*it);
		isBinary = true;
		if (it->substr(it->find_last_of(".") + 1) == "cl") {
			isBinary = false;
		}
		CL_file cl_file;
		cl_file.file = f;
		cl_file.isBinary = isBinary;
		kernel_files.push_back(cl_file);
		cout << "file loaded: " << *it << " - from " << (isBinary ? "(binary)" : "(source)") << endl;
	}
	return kernel_files;
}

/**
 * Writes binaries of a program to different files, in current directory (for now)
 */
bool saveBinaries(cl::Program program, vector<std::string> fileNames) {
	// Allocate some memory for all the kernel binary data
	const std::vector<unsigned long> binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
	std::vector<char> binData(std::accumulate(binSizes.begin(), binSizes.end(), 0));
	char *binChunk = &binData[0];


	//A list of pointers to the binary data
	std::vector<char *> binaries;
	for (unsigned int i = 0; i < binSizes.size(); ++i) {
		binaries.push_back(binChunk);
		binChunk += binSizes[i];
	}

	program.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);

	for (unsigned int i = 0; i < binaries.size() && i < fileNames.size(); ++i) {
		fileNames[i] += ".bin";
		std::ofstream binaryfile(fileNames[i].c_str(), std::ios::binary);
		binaryfile.write(binaries[i], binSizes[i]);
	}
	return true;
}

OpenCLaccelThread::OpenCLaccelThread(vine_pipe_s *v_pipe, AccelConfig &conf) : accelThread(v_pipe, conf) {
	// cout << "init_params " << conf.init_params << endl;
	// cout << "init_params " << atoi(conf.init_params.c_str()) << endl;
	this->pciId = atoi(conf.init_params.c_str());

	/* Files must be inside diamonds:
	 * - split by commas
	 * - no spaces
	 * - format is key:value mapping to file:function
	 * - each file should only have the 1 function specified
	 * Example: <vec_add.cl:vadd,vec_mul.c:vmul>
	 * loads files vec_add.cl, vec_mul.cl and functions vadd, vmul respectively
	 */
	size_t start_pos = conf.init_params.find('<') + 1;
	size_t end_pos = conf.init_params.find('>') - start_pos;

	if (start_pos != string::npos && end_pos != string::npos) {
		// std::map<std::string, std::string> files_funcs = decodeArgs(conf.init_params.substr(start_pos, end_pos));
		vector<string> files_funcs = decodeFilesArgs(conf.init_params.substr(start_pos, end_pos));
		this->kernel_files = loadFiles(files_funcs);
	} else {
		cout << "no kernel arguments provided" << endl;
	}
}

OpenCLaccelThread::~OpenCLaccelThread() {}

bool OpenCLaccelThread::loadKernels() {
	string kernelFunctionName;
	int i = 0;
	cl::Program::Sources sources;
	vector<cl::Program::Binaries> binaries;
	vector<cl_int> binaryStatus;

	try {

		// cl::Context defaultContext = cl::Context::getDefault();
		// std::vector<cl::Device> devices = {cl::Device::getDefault()};

		cl::Program bnrProgram, srcProgram;
		vector<cl::Device> devices = {defaultDevice};

		vector<cl::Kernel> srcKernels;
		vector<cl::Kernel> bnrKernels;

		for (vector<CL_file>::iterator k_f = this->kernel_files.begin(); k_f != this->kernel_files.end(); k_f++) {
			std::pair<const char *, size_t> source = std::make_pair(k_f->file.c_str(), k_f->file.size());
			if (k_f->isBinary == true) {
				cl::Program::Binaries binary = {source};
				binaries.push_back(binary);
			} else {
				sources.push_back(source);
			}
		}

		if (!sources.empty()) {
			cl::Program srcProgram = cl::Program(defaultContext, sources);
			try {
				srcProgram.build(devices);
			} catch (cl::Error err) {
				if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
					for (cl::Device dev : devices) {
						// Check the build status
						cl_build_status status = srcProgram.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
						if (status != CL_BUILD_ERROR)
							continue;

						// Get the build log
						std::string buildlog = srcProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
						cout << "Build log: " << endl
							 << buildlog.c_str() << endl;
						// printf("Build log: \n%s\n", buildlog.c_str());
					}
					return false;
				} else {
					throw err;
				}
			}
			srcProgram.createKernels(&srcKernels);

			for (vector<cl::Kernel>::iterator src = srcKernels.begin(); src != srcKernels.end(); src++) {
				kernelFunctionName = string(src->getInfo<CL_KERNEL_FUNCTION_NAME>().c_str());
				cout << "Built: \t" << kernelFunctionName << endl;
				kernels[kernelFunctionName] = *src;
			}

			cout << endl
				 << "total kernels from source: " << srcKernels.size() << endl;
		}

		if (!binaries.empty()) {
			cout << " binaries found: " << binaries.size() << endl;
			cl::Program bnrProgram;
			size_t index = 0;
			int totalKernelsFromBinaries = 0;
			// Here we need to create a cl::Program object for each binary file we have,
			// but since we have 1 device, we can use the same context (or not ?)
			while (index < binaries.size()) {
				cl::Program::Binaries binary = binaries[index++];
				try {
					bnrProgram = cl::Program(defaultContext, devices, binary, &binaryStatus);
				} catch (cl::Error err) {
					if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
						for (cl::Device dev : devices) {
							// Check the build status
							cl_build_status status = bnrProgram.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
							if (status != CL_BUILD_ERROR)
								continue;

							// Get the build log
							std::string buildlog = bnrProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
							cout << "Build log: " << endl
								 << buildlog.c_str() << endl;
							// printf("Build log: \n%s\n", buildlog.c_str());
						}
						return false;
					} else {
						throw err;
					}
				}
				bnrProgram.createKernels(&bnrKernels);

				for (vector<cl::Kernel>::iterator bnr = bnrKernels.begin(); bnr != bnrKernels.end(); bnr++) {
					kernelFunctionName = string(bnr->getInfo<CL_KERNEL_FUNCTION_NAME>().c_str());
					// bnr->getInfo(CL_KERNEL_FUNCTION_NAME, &kernelFunctionName);
					cout << kernelFunctionName << " -binary loaded- " << endl;
					kernels[kernelFunctionName] = *bnr;
					totalKernelsFromBinaries++;
				}

				if (!binaryStatus.empty()) {
					i = 0;
					for (vector<cl_int>::iterator status = binaryStatus.begin(); status != binaryStatus.end(); status++) {
						if (*status != CL_SUCCESS)
							cout << i++ << ": did not load successfully, threw error: " << *status << endl;
					}
				}
			}
			cout << endl
				 << "total kernels from binaries: " << totalKernelsFromBinaries << endl;
		}

	} catch (cl::Error err) {
		cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;

		if (!binaryStatus.empty()) {
			i = 0;
			for (vector<cl_int>::iterator status = binaryStatus.begin(); status != binaryStatus.end(); status++) {
				if (*status != CL_SUCCESS)
					cout << i++ << ": did not load successfully, threw error: " << *status << endl;
			}
		}
		return false;
	}
	return true;
}

bool OpenCLaccelThread::getNumberOfDevices() {
	int err;
	unsigned i, j;

	try {
		vector<cl::Platform> platforms;
		// This is the only call (in cpp) that gets all the platforms
		// For this function it is wasted, since we only need the number of platforms
		cl::Platform::get(&platforms);

		/* Alternative way, using C style calls
            cl_uint platformCount;
            cl_platform_id* platformIDs;

            clGetPlatformIDs(0, NULL, &platformCount);
            clGetPlatformIDs(platformCount, platforms, NULL);

            for(cl_uint i = 0; i < platform; i++) {
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
                totalDevices += deviceCount;
            }
        */

		// Select the correct platform
		vector<cl::Device> devices;
		for (i = 0; i < platforms.size(); i++) {
			// cout << endl
			// 	 << "================================" << endl;
			// cout << "Platform[" << i << "]: " << platforms[i].getInfo<CL_PLATFORM_NAME>().c_str() << endl;
			// cout << "- Version: " << platforms[i].getInfo<CL_PLATFORM_VERSION>().c_str() << endl;
			// cout << "- Profile: " << platforms[i].getInfo<CL_PLATFORM_PROFILE>().c_str() << endl;
			// cout << "- Vendor: " << platforms[i].getInfo<CL_PLATFORM_VENDOR>().c_str() << endl;
			// cout << "- Extensions: " << platforms[i].getInfo<CL_PLATFORM_EXTENSIONS>().c_str() << endl;
			try {
				err = platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
			} catch (cl::Error err) {
				if (err.err() == CL_DEVICE_NOT_FOUND)
					continue;
				else
					throw err;
			}
			if (err == CL_SUCCESS) {
				for (j = 0; j < devices.size(); j++) {
					// cout << "----------------------" << endl;
					// cout << "  Device: " << devices[i].getInfo<CL_DEVICE_NAME>().c_str() << endl;
					// cout << "  - Type " << devices[i].getInfo<CL_DEVICE_TYPE>() << endl;
					// cout << "  - Vendor " << devices[i].getInfo<CL_DEVICE_VENDOR>().c_str() << endl;
					// cout << "  - Profile " << devices[i].getInfo<CL_DEVICE_PROFILE>().c_str() << endl;
					// cout << "  - Version " << devices[i].getInfo<CL_DEVICE_VERSION>().c_str() << endl;
					// cout << "  - Driver Version " << devices[i].getInfo<CL_DRIVER_VERSION>().c_str() << endl;
					// cout << "  - Extensions " << devices[i].getInfo<CL_DEVICE_EXTENSIONS>().c_str() << endl;
					// cout << "\t- OPENCL_C_VERSION " << devices[i].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << endl;
					// cout << endl;

					this->numberOfDevices++;
				}
				this->numberOfPlatforms++;
			}
		}
		cout << "================================" << endl;

	} catch (cl::Error err) {
		cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
		return false;
	}

	return true;
}

bool OpenCLaccelThread::initDevice(int deviceNo) {
	int err, devicesLooped = 0;
	unsigned i, j;

	try {
		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::CommandQueue command_queue;

		cl::Platform::get(&platforms);
		// Select the correct platform
		for (i = 0; i < platforms.size(); i++) {
			try {
				err = platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
			} catch (cl::Error err) {
				// Handle the possibility that a platform might not have a device by just skipping it.
				if (err.err() == CL_DEVICE_NOT_FOUND)
					continue;
			}
			if (err == CL_SUCCESS) {
				for (j = 0; j < devices.size(); j++) {
					if (devicesLooped++ == deviceNo) {
						// Set the default platform and device, so every OpenCL command
						// that does not accept a device (or platform) as an argument, uses these.

						// cl::Platform::setDefault(platforms[i]);
						// cl::Device::setDefault(devices[j]);
						defaultPlatform = platforms[i];
						defaultDevice = devices[j];

						cout << endl
							 << "================================" << endl;
						cout << "Selected Platform[" << i << "]: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>().c_str() << endl;
						cout << "- Version: " << defaultPlatform.getInfo<CL_PLATFORM_VERSION>().c_str() << endl;
						cout << "- Profile: " << defaultPlatform.getInfo<CL_PLATFORM_PROFILE>().c_str() << endl;
						cout << "- Vendor: " << defaultPlatform.getInfo<CL_PLATFORM_VENDOR>().c_str() << endl;
						cout << "- Extensions: " << defaultPlatform.getInfo<CL_PLATFORM_EXTENSIONS>().c_str() << endl;


						cout << "----------------------" << endl;
						cout << "  Selected Device[" << j << "]: " << defaultDevice.getInfo<CL_DEVICE_NAME>().c_str() << endl;
						cout << "  - Type " << defaultDevice.getInfo<CL_DEVICE_TYPE>() << endl;
						cout << "  - Vendor " << defaultDevice.getInfo<CL_DEVICE_VENDOR>().c_str() << endl;
						cout << "  - Profile " << defaultDevice.getInfo<CL_DEVICE_PROFILE>().c_str() << endl;
						cout << "  - Version " << defaultDevice.getInfo<CL_DEVICE_VERSION>().c_str() << endl;
						cout << "  - Driver Version " << defaultDevice.getInfo<CL_DRIVER_VERSION>().c_str() << endl;
						cout << "  - Extensions " << defaultDevice.getInfo<CL_DEVICE_EXTENSIONS>().c_str() << endl;
						cout << endl;

						// Do we need a member variable for selectedDevice? (no, so far)
						// this->selectedDevice = devices[j];

						// Same here
						vector<cl::Device> initDevices = {devices[j]};
						cl::Context context = cl::Context(initDevices);

						// cl::Context::setDefault(context);
						defaultContext = context;

						command_queue = cl::CommandQueue(defaultContext, defaultDevice);

						// cl::CommandQueue::setDefault(command_queue);
						defaultCommandQueue = command_queue;

						return true;
					}
				}
			}
		}
	} catch (cl::Error err) {
		cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
		return false;
	}

	return false;
}

// Perform a test copy to the device
bool OpenCLaccelThread::prepareDevice() {
	float *h_a;
	int size = sizeof(int);
	try {

		cl::Buffer d_a = cl::Buffer(defaultContext, CL_MEM_READ_ONLY, size);

		h_a = (float *)malloc(size);

		defaultCommandQueue.enqueueWriteBuffer(d_a, CL_TRUE, 0, size, &h_a);

		return true;
	} catch (cl::Error err) {
		cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
		return false;
	}
}

/*initializes the GPU accelerator*/
bool OpenCLaccelThread::acceleratorInit() {
	// int numberOfGPUS = numberOfCudaDevices();

	cout << "OpenCL acceleratorInit() " << endl;
	cout << "=================================================" << endl
		 << endl;
	// Find the number of GPUs that exist in the current node, saved in this->numberOfDevices
	// (and platforms in this->numberOfPlatforms)
	if (getNumberOfDevices() == false) {
		cout << "Failed to get devices " << endl;
		return false;
	}


	cout << "Number of platforms(with devices): " << this->numberOfPlatforms << endl;
	cout << "Number of devices: " << this->numberOfDevices << endl;
	static int GPUExistInSystem[128] = {0};

	if (!__sync_bool_compare_and_swap(&GPUExistInSystem[pciId], 0, 1)) {
		cerr << "Already initialized" << endl;
		return true;
	}
	if (pciId > this->numberOfDevices) {
		cout << "The device with id -" << pciId << "- does not exist!!" << endl;
		cout << "Please set a device (second column in .config) with id smaller than " << this->numberOfDevices << endl;
		cout << "The system wil exit..." << endl;
		return false;
	}
	/*Initilizes a specific Device*/
	if (initDevice(pciId) == false) {
		cout << "Failed to set device: " << pciId << endl;
		return false;
	}
	cout << "Device initialization done: " << pciId << endl;

	if (!this->kernel_files.empty()) {
		cout << endl
			 << "Loading and building kernels..." << endl;

		chrono::time_point<chrono::system_clock> startCompilation, endCompilation;

		startCompilation = chrono::system_clock::now();
		if (loadKernels() == false) {
			cout << "Failed to load kernels" << endl;
			return false;
		}

		endCompilation = chrono::system_clock::now();
		auto elapsedCompilation = chrono::duration_cast<chrono::nanoseconds>(endCompilation - startCompilation);
		cout << endl
			 << "Kernel Compilation complete. Time elapsed: " << elapsedCompilation.count() << " nanoseconds" << endl;
	}
	OpenCLresetFlags[pciId] = false;
	/*Prepare the device */
	if (prepareDevice() == true) {
		cout << "=====================================================" << endl
			 << endl;
	} else {
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
// OpenCL doesn't really need to release the "device", but we can flush the command_queue
void OpenCLaccelThread::acceleratorRelease() {
	try {
		defaultCommandQueue.flush();

		cout << "OPENCL: Reset device was successful." << endl;
	} catch (cl::Error err) {
		cout << "Failed to reset device" << endl;
		cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
	}

	// 	if (resetCUDADevice() == true) {
	// 		//cout << "Reset device was successful." << endl;
	// 	} else {
	// 		cout << "Failed to reset device " << endl;
	// 	}

#ifdef SAMPLE_OCCUPANCY
	stop_sampling();
#endif
}

void OpenCLaccelThread::printOccupancy() {
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
bool Host2OpenCL(vine_task_msg_s *vine_task, vector<void *> &ioHD) {
	// void *tmpIn;
	cl::Buffer *tmpIn;
	bool completed = true;
	// cudaError_t errorInputs, errorOutputs;

#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startMalloc, endMalloc, startH2D, endH2D;
	double sumMalloc_In_Out; /*Duration of malloc for all input AND output data*/
	chrono::duration<double, nano> elapsedInputMalloc;
	chrono::duration<double, nano> elapsedMemcpyH2D;
	chrono::duration<double, nano> elapsedMallocOut;
#endif
	/*Map vinedata with cuda data*/
	// map<vine_data *, void *> vineData2Cuda;
	map<vine_data *, void *> vineData2Buffer;

	utils_breakdown_advance(&(vine_task->breakdown), "cudaMalloc_Inputs");

#ifdef DATA_TRANSFER
	cout << "Number of inputs: " << vine_task->in_count << endl;
	cout << "Number of outputs: " << vine_task->out_count << endl;
#endif

	mutexOpenCLAccess.lock();

	int mallocIn;
	for (mallocIn = 0; mallocIn < vine_task->in_count; mallocIn++) {
		/* Iterate till the number of inputs*/
		if (((((vine_data_s *)vine_task->io[mallocIn].vine_data)->place) & (Both)) == HostOnly) {
			ioHD.push_back(vine_data_deref(vine_task->io[mallocIn].vine_data));
			continue;
		}
#ifdef BREAKDOWNS_CONTROLLER
		/*start timer*/
		startMalloc = chrono::system_clock::now();
#endif
		// ***************************************************************************
		// ******************************* CHANGE HERE *******************************
		// ***************************************************************************
		// errorInputs = cudaMalloc(&tmpIn, vine_data_size(vine_task->io[mallocIn].vine_data));
		try {
			tmpIn = new cl::Buffer(defaultContext, CL_MEM_READ_ONLY, vine_data_size(vine_task->io[mallocIn].vine_data));
		} catch (cl::Error err) {
			cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
			// if (errorInputs != cudaSuccess) {
			cerr << "cudaMalloc FAILED for input: " << mallocIn << endl;
			/* inform the producer that he has made a mistake*/
			vine_task->state = task_failed;
			completed = false;
			vine_data_mark_ready(vpipe_s, vine_task->io[mallocIn].vine_data);
			/*In case of failure do not continue to copy*/
			mutexOpenCLAccess.unlock();
			//usleep(10000);
			throw runtime_error("failed cudaMalloc : Reset");
			// }
		}
		// ***************************************************************************

		/* Allocate memory to the device for all the inputs*/

#ifdef BREAKDOWNS_CONTROLLER
		/*stop timer*/
		endMalloc = chrono::system_clock::now();
		elapsedInputMalloc = endMalloc - startMalloc;
#endif
		/*map between vinedata and cuda alloced data*/
		// ***************************************************************************
		// ******************************* CHANGE HERE *******************************
		// ***************************************************************************
		// vineData2Cuda[vine_task->io[mallocIn].vine_data] = tmpIn;
		vineData2Buffer[vine_task->io[mallocIn].vine_data] = tmpIn;
		// ***************************************************************************
	}

	/*End Malloc input -  Start cudaMemcpy Inputs Host to Device*/
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMemCpy_H2G");

	if (vine_task->state != task_failed) {
		int memCpyIn;
		for (memCpyIn = 0; memCpyIn < vine_task->in_count; memCpyIn++) {

#ifdef BREAKDOWNS_CONTROLLER
			/*stop timer*/
			startH2D = chrono::system_clock::now();
#endif

			// ***************************************************************************
			// ******************************* CHANGE HERE *******************************
			// ***************************************************************************
			// tmpIn = vineData2Cuda[vine_task->io[memCpyIn].vine_data];
			tmpIn = (cl::Buffer *)vineData2Buffer[vine_task->io[memCpyIn].vine_data];
			try {
				defaultCommandQueue.enqueueWriteBuffer(*tmpIn, CL_TRUE, 0, vine_data_size(vine_task->io[memCpyIn].vine_data), vine_data_deref(vine_task->io[memCpyIn].vine_data));
				// this->command_queue.enqueueWriteBuffer(tmpIn, CL_FALSE, 0, vine_data_size(vine_task->io[memCpyIn].vine_data), &vine_data_deref(vine_task->io[memCpyIn].vine_data));

			} catch (cl::Error err) {
				cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
				/* Copy inputs to the device */
				// if (cudaMemcpy(tmpIn, vine_data_deref(vine_task->io[memCpyIn].vine_data), vine_data_size(vine_task->io[memCpyIn].vine_data), cudaMemcpyHostToDevice) != cudaSuccess) {
				cerr << "Cuda Memcpy (Host2Device) FAILED for input: " << memCpyIn << endl;
				vine_task->state = task_failed;
				completed = false;
				vine_data_mark_ready(vpipe_s, vine_task->io[memCpyIn].vine_data);
				mutexOpenCLAccess.unlock();
				throw runtime_error("failed cudaMemcpy : Reset");
				// }
			}
			// ***************************************************************************


#ifdef DATA_TRANSFER
			cout << "Size of input " << memCpyIn << " is: " << vine_data_size(vine_task->io[memCpyIn].vine_data) << endl;
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
		for (out = mallocIn; out < vine_task->out_count + mallocIn; out++) {
#ifdef BREAKDOWNS_CONTROLLER
			/*start timer*/
			startMalloc = chrono::system_clock::now();
#endif

			// ***************************************************************************
			// ******************************* CHANGE HERE *******************************
			// ***************************************************************************
			if (((vine_data_s *)(vine_task->io[out].vine_data))->flags & VINE_INPUT) {
				// tmpOut = vineData2Cuda[vine_task->io[out].vine_data];
				tmpOut = vineData2Buffer[vine_task->io[out].vine_data];
			} else {
				// errorOutputs = cudaMalloc(&tmpOut, vine_data_size(vine_task->io[out].vine_data));
				try {
					tmpOut = new cl::Buffer(defaultContext, CL_MEM_WRITE_ONLY, vine_data_size(vine_task->io[out].vine_data));
				} catch (cl::Error err) {
					cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
					// if (errorOutputs != cudaSuccess) {
					cerr << "cudaMalloc FAILED for output: " << out << endl;
					vine_task->state = task_failed;
					completed = false;
					vine_data_mark_ready(vpipe_s, vine_task->io[out].vine_data);
					mutexOpenCLAccess.unlock();
					throw runtime_error("failed cudaMalloc : Reset");
					// }
					//cudaMemset(tmpOut, 0, vine_data_size(vine_task->io[out].vine_data));
				}
			}
			// ***************************************************************************

#ifdef BREAKDOWNS_CONTROLLER
			/*stop timer*/
			endMalloc = chrono::system_clock::now();
			elapsedMallocOut = endMalloc - startMalloc;
#endif

			/*End cudaMalloc for outputs - Start Kernel Execution time*/
			ioHD.push_back(tmpOut);

#ifdef BREAKDOWNS_CONTROLLER
			/*malloc duration of all inputs + outputs*/
			sumMalloc_In_Out = elapsedMallocOut.count() + elapsedInputMalloc.count();
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

	mutexOpenCLAccess.unlock();
	return completed;

	return true;
}

/* Cuda Memcpy from Device to host*/
bool OpenCL2Host(vine_task_msg_s *vine_task, vector<void *> &ioDH) {
	int out;
	bool completed = true;

	/*Stop Kernel Execution time - Start cudaMemCpy for outputs Device to Host*/
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMemCpy_G2H");

#ifdef BREAKDOWNS_CONTROLLER
	// TODO: ADD openCLDeviceSynchronize();
	// cudaDeviceSynchronize();
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startD2H, endD2H;
	/*start timer*/
	startD2H = chrono::system_clock::now();
#endif

	mutexOpenCLAccess.lock();

	for (out = vine_task->in_count; out < vine_task->out_count + vine_task->in_count; out++) {

#ifdef DATA_TRANSFER
		cout << "Size of output " << out << " is: " << vine_data_size(vine_task->io[out].vine_data) << endl;
#endif

		try {

			cl::Buffer *tmp_deref = (cl::Buffer *)ioDH[out];
			//cl::Buffer *tmp_deref = (cl::Buffer *)vine_data_deref(vine_task->io[out].vine_data);
			// cl::Buffer *tmp_ioDH = (cl::Buffer *)ioDH[out];<
			defaultCommandQueue.enqueueReadBuffer(*tmp_deref, CL_TRUE, 0, vine_data_size(vine_task->io[out].vine_data), vine_data_deref(vine_task->io[out].vine_data));
			// cl::enqueueWriteBuffer(*tmp_deref, CL_TRUE, 0, vine_data_size(vine_task->io[out].vine_data), ioDH[out]);
		} catch (cl::Error err) {
			cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
			/* Copy inputs to the device */
			// if (cudaMemcpy(vine_data_deref(vine_task->io[out].vine_data), ioDH[out], vine_data_size(vine_task->io[out].vine_data), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cerr << "Cuda Memcpy (Device2Host) FAILED for output: " << out << endl;
			vine_task->state = task_failed;
			completed = false;
			vine_data_mark_ready(vpipe_s, vine_task->io[out].vine_data);
			mutexOpenCLAccess.unlock();
			throw runtime_error("failed cudaMemcpy : Reset");
			// }
			return false;
		}

		completed = true;
		if (out == vine_task->out_count + vine_task->in_count - 1) {
			vine_task->state = task_completed;
			/*End cudaMemCpy for outputs Device to Host - Start cudaMemFree*/
			utils_breakdown_advance(&(vine_task->breakdown), "cudaMemFree");
		}
		vine_data_mark_ready(vpipe_s, vine_task->io[out].vine_data);
	}
#ifdef BREAKDOWNS_CONTROLLER
	/*stop timer*/
	endD2H = chrono::system_clock::now();
	/*duration*/
	chrono::duration<double, nano> elapsed_D2H = endD2H - startD2H;
	cout << "cudaMemcpy D2H (outputs): " << elapsed_D2H.count() << " nanosec." << endl;
#endif
	mutexOpenCLAccess.unlock();
	return completed;

	return true;
}

/* Free Device memory */
bool OpenCLMemFree(vector<void *> &io) {
	// cudaError_t errorFree;
#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startFree, endFree;
	/*start timer*/
	startFree = chrono::system_clock::now();
#endif
	mutexOpenCLAccess.lock();

	bool completed = true;
	set<void *> unique_set(io.begin(), io.end());
	for (set<void *>::iterator itr = unique_set.begin(); itr != unique_set.end(); itr++) {

		try {
			delete (cl::Buffer *)*itr;
			// Not present in c++ wrapper
			// cl::clReleaseMemObject(*itr);

			// errorFree = cudaFree(*itr);
		} catch (cl::Error err) {
			cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
			// if (errorFree != cudaSuccess) {
			cerr << "cudaFree FAILED " << endl;
			completed = false;
			mutexOpenCLAccess.unlock();
			throw runtime_error("failed cudaFree : Reset");
			// }
		}
	}

#ifdef BREAKDOWNS_CONTROLLER
	/*stop timer*/
	endFree = chrono::system_clock::now();
	/*duration*/
	chrono::duration<double, nano> elapsed_Free = endFree - startFree;
	cout << "Free took : " << elapsed_Free.count() << " nanosec" << endl;
	cout << "------------------ End Breakdown ----------------" << endl;
#endif

	mutexOpenCLAccess.unlock();
	return completed;

	return true;
}

extern cl::Kernel OpenCLGetKernel(string name) {
	return kernels[name];
}

//bool shouldResetGpu (int device)
bool shouldResetOpenCL() {
	// int device;
	// //	if (device == -1)
	// cudaGetDevice(&device);

	// //return ( OpenCLresetFlags[device].exchange(false) );
	// return (OpenCLresetFlags.at(device).exchange(false));

	return true;
}

extern bool resetPolicy;

// cudaError_t my_cudaDeviceSynchronize() __attribute__((used));
// cudaError_t my_cudaDeviceSynchronize() {
// 	int device;
// 	cudaGetDevice(&device);


// 	if (!resetPolicy) {
// 		return cudaDeviceSynchronize();
// 	}

// 	//cerr << __func__ << endl;
// 	cudaError_t err = cudaSuccess;
// 	cudaEvent_t kernelFinished;

// 	err = cudaEventCreate(&kernelFinished);
// 	if (err != cudaSuccess) {
// 		fprintf(stderr, "Cuda error (cudaEventCreate): %s GPUid: %d.\n", cudaGetErrorString(err), device);
// 		return err;
// 	}
// 	err = cudaEventRecord(kernelFinished);
// 	if (err != cudaSuccess) {
// 		fprintf(stderr, "Cuda error (cudaEventRecord): %s GPUid: %d.\n", cudaGetErrorString(err), device);
// 		return err;
// 	}

// 	while ((err = cudaEventQuery(kernelFinished)) == cudaErrorNotReady) {
// 		usleep(1000);

// 		if (shouldResetGpu()) {
// 			cerr << "> Actual Reset!! GPU " << device << endl;
// 			cudaDeviceReset();
// 			//cudaDeviceSynchronize();
// 			return err;
// 		}
// 	}
// 	return err;
// }

void OpenCLaccelThread::reset(accelThread *caller) {
	// int jp = 2;
	// const char * str[3] = {"BatchJobs","UserJobs","Idle"};
	// auto runningTask = getAccelConfig().accelthread->getRunningTask();

	// if (runningTask)
	// {
	// 	jp = vine_vaccel_get_job_priority((vine_vaccel_s *)(runningTask->accel));
	// }

	// cerr << "NOW" << "(" << getAccelConfig().name << "," << str[jp] << ")"<<" Caller: "<<caller->getAccelConfig().name << endl;

	// OpenCLresetFlags.at(pciId) = true;
}

REGISTER_ACCEL_THREAD(OpenCLaccelThread)
