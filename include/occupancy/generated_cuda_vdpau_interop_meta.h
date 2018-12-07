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
// This file is generated.  Any changes you make will be lost during the next clean build.

// CUDA public interface, for type definitions and api function prototypes
#include "cuda_vdpau_interop.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Currently used parameter trace structures 
typedef struct cudaVDPAUGetDevice_v3020_params_st {
    int *device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} cudaVDPAUGetDevice_v3020_params;

typedef struct cudaVDPAUSetVDPAUDevice_v3020_params_st {
    int device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} cudaVDPAUSetVDPAUDevice_v3020_params;

typedef struct cudaGraphicsVDPAURegisterVideoSurface_v3020_params_st {
    struct cudaGraphicsResource **resource;
    VdpVideoSurface vdpSurface;
    unsigned int flags;
} cudaGraphicsVDPAURegisterVideoSurface_v3020_params;

typedef struct cudaGraphicsVDPAURegisterOutputSurface_v3020_params_st {
    struct cudaGraphicsResource **resource;
    VdpOutputSurface vdpSurface;
    unsigned int flags;
} cudaGraphicsVDPAURegisterOutputSurface_v3020_params;

// Parameter trace structures for removed functions 


// End of parameter trace structures
