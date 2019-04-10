Controller that handles and serves vine_talk requests
# Folder layout
* include - Header files that implement CPU, GPU and OpenCL methods, and the library for the exitsing libraries.
            1. VineLibMgr.h : contains the specifications for the array inside each ".so" file.
            2. VineLibUtilsCPU.h, VineLibUtilsGPU.h or VineLibUtilsOpenCL.h: contains the implementation of the functions that are used in order to transfer data to the accelerator memory. 
* src - Source code - for the controller
            1. vine_controller.cpp: contains the implementation of the controller.
            2. VineLibMgr.cpp: the implementation of vine library manager array.

# Building
* First, build vine\_talk according to its instructions.
* Then create a directory build (mkidir build)
	* Inside (cd build) build type ccmake ../
	* Without modifing anything type c and then g
	* Then do make  
* cmake automaticaly checks the existance of cuda, in both cases (with, without cuda) it creates one executable named as vine_controller that supports both situations.

# Run
There are two configuration files named as example_config and customerWeights:

##  customerWeights
This file should be created only in the case that you have selected WeightedRoundRobin (enabled On in Cmake) in the building procedure of vine_controler. This filee contains a list with the Weights of the customers in the system, this weights are between 1 and 100.

## example_config
The example_config: is used firstly for specifying the path that the dynamic libraries of the supported applications are located(path). Secondly, is used for specifying the available accelerators in the system. In the second column you should add the accelererator type (GPU, CPU, FPGA etc), in the third you specify the name of the accelerator. In the fourth column the core of the system that is responsible for that accelerator is depicted. This column is used from steafinity, to run a thread in a specific core in order to avoid sharing. Finally the last column represents the place of the accelerator, for instance in case of a GPU accelerator it depicts the PCI id.
<br/>
<br/>
In the case where an **OpenCL** accelerator is used, there are some additional points:
1. The selection of the device depends on the number of devices in the system that support OpenCL run-time. For the time being, it is a number, 0-(no. of accelerators), that needs to be decided by testing each one.
2. An extra argument is used that contains the absolute (or relative) paths for each OpenCL kernel file (either source (.cl), or a binary) to be used by the applications. Please not that source files (.cl) can only be used by GPUs, as FPGAs need a compiled binary, compiled from source by each vendors provided compiler.
<br/>

Additionaly, is used for making groups of different accelerators and adding a specifc scheduler to each of those groups. In more details:

This file contains :<br />
	1. The library path of the applications. In other words the directory that contains the dynamic libraries of the applications (../vine_applications).<br />
	2. The accelerators that exist in the current node. The format for the accelerators is the following

             Accelerator Type    Accelerator Name     Core to place accelerator thread    Accelerator selection (PCI id)    Argument(s)
    accel          GPU               GPU0                            0                               0                  
    accel          CPU               CPU0                            1                               -        
    accel          OpenCL            OPENCL0                         2                               0                      <"../../kernels/kernel1.cl", "../../kernels/kernel2.bin">

<br />
Row 1 :From the above example the system has a GPU  accelerator , the accelerator thread will run in the core 0 (Htop will show core 1), the accelerator name is GPU0, and the accelerator thread will serve the device which is in PCI 0. 
<br />
Row 2 :From the above example the system has also a CPU accelerator , the accelerator thread will run in the core 1 (Htop will show core 2), the accelerator name is CPU0.
<br />
Row 3 :From the above example the system has also an accelerator that supports OpenCL run-time. The accelerator will run in core 2 (Htop will show core 3), the accelerator name is OPENCL0. The controller will use 2 OpenCL kernel files. The first kernel is from a source file (.cl), the second is an already compiled binary.

<br />
    3. The accelerators that compose an accelGroup.
        This option is specified by the key word "group". The first column represents the name of the group, the second the scheduler of that group and then the accelerators that compose that group.
        
		       GroupName     GroupScheduler     AcceleratorsOfGroup
          group     Fast_GPUs       weighted1*           GPU0 GPU1                
          group     Slow_GPUs       weighted2*           GPU2

  <br />        
    4. Scheduler choice per accelGroup. The schedulers implemented are the WeightedRoundRoubin and VineRoundRobin. In our system each group of accelerators can use a different a scheduler. The choices are WeightedRoundRobin and VineRoundRobin 
    
                   SchedulerName        Scheduler
          sched     *weighted1     WeightedRoundRobin        
          sched     *weighted2     WeightedRoundRobin
<br />

## Execute
As soon as the ".so" files are create when one run the controller with the path to the files, the controller knows which kernels to execute.
./build/vine_controller "example_config"


#  Creating libraries for the Controller

## Create ".so" for CPU
1. Create a file that will contains the kernel, the dispatcher function and a table that contains all the supported functions for this specific ".so" (vine-applications/src/c_darkGray.cpp).
   
    1.a. Create a ".h" file that has a reference to the function that is going to be implemented. (eg in vine-applications/include/c_darkGray.h). Include the .h in the cpp file that is going to be used for the ".so" (vine-applications/src/c_darkGray.cpp ) .
    
    1.b. The kernel is the function that is going to be executed in the accelerator (in vine-applications/src/c_darkGray.cpp is void darkGray(...)).
    
    1.c. The dispatcher function (hostCode) contains; the memory allocation for inputs and outputs and data trasfer from host to device memory (Host2CPU), the kernel call (darkGray (args, inputs, outputs)), the data transfer from device to host memory (CPU2Host), and the deallocation of memory in the device (CPUMemFree). For CPU as accelerator functions like Host2CPU,  CPU2Host, and CPUMemFree does not transfer anything due to the fact that CPU uses host memory, they are used in such a manner for homogenity. 
    
    1.d. the table contains all the functions supported from this .so . In the c_darkGray.cpp lines VINE_PROC_LIST_START(), VINE_PROCEDURE("darkGray", CPU, hostCode, sizeof(darkGrayArgs)), and VINE_PROC_LIST_END() create this table. In VINE_PROCEDURE one should add the following ("function name", "accelerator type", "the dispatcher function name", "argument size")
    
2. Type the following in order to create ".so" library that contains the kernel : g++ -fPIC -shared <your_function_name.cpp> -o  </path/to/your_function_name.so> (eg g++ -fPIC -shared c_darkGray.cpp -o ../lib/c_darkGray.so) .

## Create ".so" GPU

1. Create a file that will contains the kernel, the dispatcher function and a table that contains all the supported functions for this specific ".so" (vine-applications/src/cu_darkGray.cu).
    
    1.a. Create a ".h" file that has a reference to the function that is going to be implemented. (eg in vine-applications/include/cu_darkGray.h). Include the .h in the cu file that is going to be used for the ".so" (vine-applications/src/cu_darkGray.cu ) .
    
    1.b. The kernel is the function that is going to be executed in the accelerator (in vine-applications/src/cu_darkGray.cu is __global__ void rgb_gray(...). It is called from function cu_darkGray(...)
    
    1.c. The dispatcher function (hostCode) contains; the memory allocation for inputs and outputs and data trasfer from host to device memory (Host2GPU), the kernel call (cu_darkGra (args, inputs, outputs)), the data transfer from device to host memory (GPU2Host), and the deallocation of memory in the device (GPUMemFree).  
    
    1.d. the table contains all the functions supported from this .so . In the c_darkGray.cpp lines VINE_PROC_LIST_START(), VINE_PROCEDURE("darkGray", GPU, hostCode, sizeof(darkGrayArgs)), and VINE_PROC_LIST_END() create this table. In VINE_PROCEDURE one should add the following ("function name", "accelerator type", "the dispatcher function name", "argument size")
    
2. Type the following in order to create ".so" library that contains the kernel : g++ -fPIC -shared <your_function_name.cpp> -o  </path/to/your_function_name.so> (eg g++ -fPIC -shared c_darkGray.cpp -o ../lib/c_darkGray.so) .

## Create ".so" OpenCL

1. Create a file that will contains the kernel, the dispatcher function and a table that contains all the supported functions for this specific ".so" (vine-applications/src/opencl_darkGray.cpp).
    
    1.a. Create a ".h" file that has a reference to the function that is going to be implemented. (eg in vine-applications/include/opencl_darkGray.h). Include the .h in the .cpp file that is going to be used for the ".so" (vine-applications/src/opencl_darkGray.cpp ) .

    1.b At the top of the (vine-applications/src/opencl_darkGray.cpp) file this declaration needs to be placed, in order to be able to execute the cl::Kernel object: `extern cl::CommandQueue defaultCommandQueue;`. 
    
    1.c. The kernel is the function that is going to be executed in the accelerator (in vine-applications/kernels/darkGray.cl is __kernel void rgb_gray(...). It is called from function opencl_darkGray(...). To get the cl::Kernel object that contains the (compiled) kernel function, the function `cl::Kernel kernel = OpenCLGetKernel(kernel_name_string)` has to be used.

    1.d. The dispatcher function (hostCode) contains; the memory allocation for inputs and outputs and data trasfer from host to device memory (Host2OpenCL), the kernel call (opencl_darkGray(args, inputs, outputs)), the data transfer from device to host memory (OpenCL2Host), and the deallocation of memory in the device (OpenCLMemFree).
    
    1.e. the table contains all the functions supported from this .so . In the vine_darkGray.cpp lines VINE_PROC_LIST_START(), VINE_PROCEDURE("darkGray", OPEN_CL, hostCode, sizeof(darkGrayArgs)), and VINE_PROC_LIST_END() create this table. In VINE_PROCEDURE one should add the following ("function name", "accelerator type", "the dispatcher function name", "argument size")
    
2. Type the following in order to create ".so" library that contains the kernel : `g++ -fPIC -shared -lOpenCL <your_function_name.cpp> -o  </path/to/your_function_name.so>` ( eg `g++ -fPIC -shared -lOpenCL c_darkGray.cpp -o ../lib/c_darkGray.so` ) .

# Notes
* If by mistake one run first an application and then the controller, he/she will get segmentation fault. In order to solve it please delete the share segment by typing rm /dev/shm/"shm name" and run the controller firstly and then the application.
* (WARNING) In case that you create a config file that the first row is "GPU 0 GPU0" and the second "CPU 0 CPU0", the thread responsible for both accelerators will run to core0. As a result it will decrease the overall performance. So be very carefull in the config creation.

