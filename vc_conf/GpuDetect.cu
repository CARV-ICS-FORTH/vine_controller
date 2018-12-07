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
	#include "GpuDetect.h"
#include <sstream>
#include <fstream>
#include <regex>

std::vector<Accel> getGpus(bool verbose)
{
	std::vector<Accel> gpus;
	int gpu_n = 0;
	Accel temp;

	cudaGetDeviceCount(&gpu_n);

	for(int gpu = 0 ; gpu < gpu_n ; ++gpu)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, gpu);
		temp.type = "GPU";
		temp.name = "GPU";
		temp.name += std::to_string(gpu);
		temp.id = gpu;

		temp.com = prop.name; 

		std::ostringstream oss;

		oss << "/sys/class/pci_bus/0000:";
		oss.width(2);
		oss.fill('0');
		oss  << std::hex << prop.pciBusID << "/cpuaffinity";

		if(verbose)
		{
			temp.com += " (";
			temp.com += "asyncEngineCount:" + std::to_string(prop.asyncEngineCount) + ", ";
			temp.com += "maxThreadsPerBlock:" + std::to_string(prop.maxThreadsPerBlock) + ", ";
			temp.com += "maxThreadsPerMultiProcessor:" + std::to_string(prop.maxThreadsPerMultiProcessor) + ", ";
			temp.com += "multiProcessorCount:" + std::to_string(prop.multiProcessorCount) + ", ";
			temp.com += "unifiedAddressing:" + std::to_string(prop.unifiedAddressing) + ", ";
			temp.com += "affinity:" + oss.str();
			temp.com += ")";
		}


		std::ifstream sys(oss.str().c_str());

		std::string raw;

		sys >> raw;

		// Remove , from bitstring
		size_t found;
		while(std::string::npos != (found=raw.find_first_of(",")) )
		{
			raw = raw.substr(0,found) + raw.substr(found+1);
		}

		unsigned long long map;

		std::istringstream iss(raw);

		iss>> std::hex >>map;

		temp.cpumask = map;

		gpus.push_back(temp);

		if(verbose)
		{
			if(gpu == 0)
			{
				std::cerr << "# P2P support\n";
				std::cerr << "# X ";
				for(int cnt = 0 ; cnt < gpu_n ; cnt++)
				{
					std::cerr.width(2);
					std::cerr << cnt;
				}
				std::cerr << std::endl;
			}
			std::cerr << "#";
			std::cerr.width(2);
			std::cerr << gpu << ' ';
			for(int other = 0 ; other < gpu_n ; other++)
			{
				int t;

				cudaDeviceCanAccessPeer(&t,gpu,other);
				std::cerr.width(2);
				std::cerr << t;
			}
			std::cerr << std::endl;
		}
	}
	
	return gpus;
}
