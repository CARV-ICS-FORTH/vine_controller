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
#include <string>
#include <unistd.h>
#include "GpuDetect.h"

void help(const char * exe)
{
	std::cerr << "Usage: " << exe;
	std::cerr << " -c <cpus> -v -h" << std::endl << std::endl;
	std::cerr << "\t-c <cpu>\tSet number of cpus." << std::endl;
	std::cerr << "\t-v\t\tVerbose gpu output." << std::endl;
	std::cerr << "\t-h\t\tThis help messages."  << std::endl;
	std::cerr << std::endl;
}

int main(int argc,char * argv[])
{
	int c;
	int cpus = 0;
	bool verbose = false;

	while((c = getopt (argc, argv, "c:vh")) != -1)
	{
		switch(c)
		{
			case 'v':
				verbose = true;
				break;
			case 'c':
				cpus = atoi(optarg);
				break;
			case 'h':
			default:
				help(argv[0]);
				return -1;
				break;
		}
	}

	std::vector<Accel> gpus = getGpus(verbose);
	std::bitset<64> available_cores(-1);

	std::cout << "# Detected " << gpus.size() << " gpus" << std::endl;

	for(auto gpu : gpus)
	{
		gpu.pin(available_cores);
		std::cout << gpu;
	}

	std::cout << "# Created " << cpus << " cpus" << std::endl;
	for(int cpu = 0 ; cpu < cpus ; cpu++)
	{
		Accel c;
		c.type = "CPU";
		c.name = "CPU";
                c.name += std::to_string(cpu);
		c.pin(available_cores);
		 std::cout << c;
	}

	return 0;
}
