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

// Dependent includes
#include <vdpau/vdpau.h>

// CUDA public interface, for type definitions and cu* function prototypes
#include "cudaVDPAU.h"


// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct cuVDPAUGetDevice_params_st {
    CUdevice *pDevice;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} cuVDPAUGetDevice_params;

typedef struct cuVDPAUCtxCreate_v2_params_st {
    CUcontext *pCtx;
    unsigned int flags;
    CUdevice device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} cuVDPAUCtxCreate_v2_params;

typedef struct cuGraphicsVDPAURegisterVideoSurface_params_st {
    CUgraphicsResource *pCudaResource;
    VdpVideoSurface vdpSurface;
    unsigned int flags;
} cuGraphicsVDPAURegisterVideoSurface_params;

typedef struct cuGraphicsVDPAURegisterOutputSurface_params_st {
    CUgraphicsResource *pCudaResource;
    VdpOutputSurface vdpSurface;
    unsigned int flags;
} cuGraphicsVDPAURegisterOutputSurface_params;

typedef struct cuVDPAUCtxCreate_params_st {
    CUcontext *pCtx;
    unsigned int flags;
    CUdevice device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} cuVDPAUCtxCreate_params;
