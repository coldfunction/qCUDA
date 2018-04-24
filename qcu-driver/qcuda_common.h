
#ifndef QCUDA_COMMON_H
#define QCUDA_COMMON_H

////////////////////////////////////////////////////////////////////////////////
///	common variables
////////////////////////////////////////////////////////////////////////////////

#define QCU_KMALLOC_SHIFT_BIT 22
#define QCU_KMALLOC_MAX_SIZE (1UL<<QCU_KMALLOC_SHIFT_BIT)
// don't bigger than 1<<22

#if 0
#define USER_KERNEL_COPY
#else
#endif

#define VIRTIO_ID_QC 69

enum
{
	VIRTQC_CMD_WRITE = 100,
	VIRTQC_CMD_READ,
	VIRTQC_CMD_OPEN,
	VIRTQC_CMD_CLOSE,
	VIRTQC_CMD_MMAP,
	VIRTQC_CMD_MUNMAP,
	VIRTQC_CMD_MMAPCTL,
	VIRTQC_CMD_MMAPRELEASE,
	
};

enum
{
	// Module & Execution control (driver API)
	VIRTQC_cudaRegisterFatBinary = 200,
	VIRTQC_cudaUnregisterFatBinary,
	VIRTQC_cudaRegisterFunction,
	VIRTQC_cudaRegisterVar,
	VIRTQC_cudaLaunch,
    VIRTQC_cudaFuncGetAttributes,

	// Memory Management (runtime API)
	VIRTQC_cudaMalloc,
	VIRTQC_cudaMemcpy,
	VIRTQC_cudaMemcpyAsync,
	VIRTQC_cudaMemset,
	VIRTQC_cudaFree,

	// Device Management (runtime API)
	VIRTQC_cudaGetDevice,
	VIRTQC_cudaGetDeviceCount,
    VIRTQC_cudaDeviceGetAttributes,
	VIRTQC_cudaGetDeviceProperties,
	VIRTQC_cudaSetDevice,
	VIRTQC_cudaDeviceSynchronize,
	VIRTQC_cudaDeviceReset,
	VIRTQC_cudaDeviceSetLimit,
    VIRTQC_cudaDeviceSetCacheConfig,

	// Version Management (runtime API)
	VIRTQC_cudaDriverGetVersion,
	VIRTQC_cudaRuntimeGetVersion,

	// Event Management (runtime API)
	VIRTQC_cudaEventCreate,
	VIRTQC_cudaEventCreateWithFlags,
	VIRTQC_cudaEventRecord,
	VIRTQC_cudaEventSynchronize,
	VIRTQC_cudaEventElapsedTime,
	VIRTQC_cudaEventDestroy,

	// Error Handling (runtime API)
	VIRTQC_cudaGetLastError,
    VIRTQC_cudaPeekAtLastError,

	//zero-cpy
	VIRTQC_cudaHostRegister,
	VIRTQC_cudaHostGetDevicePointer,
	VIRTQC_cudaHostUnregister,
	VIRTQC_cudaSetDeviceFlags,
	VIRTQC_cudaFreeHost,

	//stream
	VIRTQC_cudaStreamCreate,
	VIRTQC_cudaStreamDestroy,
	VIRTQC_cudaStreamSynchronize,
    VIRTQC_cudaStreamWaitEvent,

	// Thread Management
	VIRTQC_cudaThreadSynchronize,
};

typedef struct VirtioQCArg   VirtioQCArg;

// function args
struct VirtioQCArg
{
	int32_t cmd;
	uint64_t rnd;
	uint64_t para;
	
	uint64_t pA;
	uint32_t pASize;

	uint64_t pB;
	uint32_t pBSize;

	uint32_t flag;

};

#endif
