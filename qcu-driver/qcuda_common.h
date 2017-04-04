
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
	VIRTQC_CMD_MMAPRELEASE,
	VIRTQC_CMD_MMAPCTL
};

enum
{
	// Module & Execution control (driver API)
	VIRTQC_cudaLaunch = 200,
	VIRTQC_cudaRegisterFunction,
	VIRTQC_cudaRegisterFatBinary,
	VIRTQC_cudaUnregisterFatBinary,

	// Memory Management (runtime API)
	VIRTQC_cudaMalloc,
	VIRTQC_cudaMemcpy,
	VIRTQC_cudaMemcpyAsync,
	VIRTQC_cudaMemset = 208,
	VIRTQC_cudaFree,

	// Device Management (runtime API)
	VIRTQC_cudaGetDevice, // 210
	VIRTQC_cudaGetDeviceCount = 212,
	VIRTQC_cudaDeviceReset,
	VIRTQC_cudaSetDevice,
	VIRTQC_cudaDeviceSynchronize = 216, // 213 no args
	VIRTQC_cudaGetDeviceProperties,

	// Version Management (runtime API)
	VIRTQC_cudaDriverGetVersion,
	VIRTQC_cudaRuntimeGetVersion = 220,

	// Event Management (runtime API)
	VIRTQC_cudaEventCreate = 180,
	VIRTQC_cudaEventCreateWithFlags = 244, //244,
	VIRTQC_cudaEventRecord = 164,
	VIRTQC_cudaEventSynchronize = 246,
	VIRTQC_cudaEventElapsedTime = 248, //190
	VIRTQC_cudaEventDestroy = 228,

	// Error Handling (runtime API)
	VIRTQC_cudaGetLastError,

	//zero-cpy
	VIRTQC_cudaHostRegister,
	VIRTQC_cudaHostGetDevicePointer = 232,
	VIRTQC_cudaHostUnregister,
	VIRTQC_cudaSetDeviceFlags = 160,
	VIRTQC_cudaFreeHost = 236,

	//stream
	VIRTQC_cudaStreamCreate = 240,
	VIRTQC_cudaStreamDestroy, // 238
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
