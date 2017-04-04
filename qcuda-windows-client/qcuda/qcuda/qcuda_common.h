#pragma once

#ifndef QCUDA_COMMON_H
#define QCUDA_COMMON_H
#include <initguid.h>

#define DEV_NAME L"\\Device\\qcuda"
#define DEV_SYMBOLIC_LINK L"\\DosDevices\\qcuda"
#define DEV_PATH L"\\DosDevices\\C:\\qcuda"
#define DEV_FILE_PATH L"\\DosDevices\\C:\\qcuda\\qcuda.txt"
#define POOL_TAG 'ELIF'

DEFINE_GUID(GUID_DEVINTERFACE_qcudriver,
	0x78A1C341, 0x4539, 0x11d3, 0xB8, 0x8D, 0x00, 0xC0, 0x4F, 0xAD, 0x51, 0x71);

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

/*
/// Module & Execution control (driver API)
#define VIRTQC_cudaRegisterFatBinary CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8001, METHOD_IN_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaUnregisterFatBinary CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8002, METHOD_IN_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaRegisterFunction CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8003, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaLaunch CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8004, METHOD_IN_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
//
// Memory Management (runtime API)
#define VIRTQC_cudaMalloc CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8005, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaMemcpy CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8006, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaMemcpyAsync CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8007, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaMemset CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8008, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaFree CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8009, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
//
// Device Management (runtime API)
#define VIRTQC_cudaGetDevice CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800A, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaGetDeviceCount CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800B, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaGetDeviceProperties CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800C, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaSetDevice CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800D, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaDeviceSynchronize CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800E, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaDeviceReset CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800F, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
//
// Version Management (runtime API)
#define VIRTQC_cudaDriverGetVersion CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8010, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaRuntimeGetVersion CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8011, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
//
// Event Management (runtime API)
#define VIRTQC_cudaEventCreate CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8012, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaEventCreateWithFlags CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8013, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaEventRecord CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8014, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaEventSynchronize CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8015, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaEventElapsedTime CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8016, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaEventDestroy CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8017, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
//
// Error Handling (runtime API)
#define VIRTQC_cudaGetLastError CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8018, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
//
////zero-cpy
#define VIRTQC_cudaHostRegister CTL_CODE(FILE_DEVICE_UNKNOWN, 0x8019, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaHostGetDevicePointer CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801A, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaHostUnregister CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801B, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaSetDeviceFlags CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801C, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaFreeHost CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801D, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
//
////stream
#define VIRTQC_cudaStreamCreate CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801E, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
#define VIRTQC_cudaStreamDestroy CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801F, METHOD_OUT_DIRECT, FILE_READ_DATA | FILE_WRITE_DATA)
*/

//******************************************************************************************
/*
enum
{
	// Module & Execution control (driver API)
	VIRTQC_cudaRegisterFatBinary = 2277381,
	VIRTQC_cudaUnregisterFatBinary,
	VIRTQC_cudaRegisterFunction,
	VIRTQC_cudaLaunch,
									
	// Memory Management (runtime API)
	VIRTQC_cudaMalloc,
	VIRTQC_cudaMemcpy,
	VIRTQC_cudaMemcpyAsync,
	VIRTQC_cudaMemset,
	VIRTQC_cudaFree,

	// Device Management (runtime API)
	VIRTQC_cudaGetDevice,
	VIRTQC_cudaGetDeviceCount,
	VIRTQC_cudaGetDeviceProperties,
	VIRTQC_cudaSetDevice,
	VIRTQC_cudaDeviceSynchronize,
	VIRTQC_cudaDeviceReset,

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

	//zero-cpy
	VIRTQC_cudaHostRegister,
	VIRTQC_cudaHostGetDevicePointer,
	VIRTQC_cudaHostUnregister,
	VIRTQC_cudaSetDeviceFlags,
	VIRTQC_cudaFreeHost,

	//stream
	VIRTQC_cudaStreamCreate,
	VIRTQC_cudaStreamDestroy

}; */
//******************************************************************************************

enum
{
	// Module & Execution control (driver API)
	VIRTQC_cudaLaunch = 200,
	VIRTQC_cudaRegisterFunction,
	VIRTQC_cudaRegisterFatBinary,
	VIRTQC_cudaUnregisterFatBinary, // 203 no args

	// Memory Management (runtime API)
	VIRTQC_cudaMalloc,
	VIRTQC_cudaMemcpy,
	VIRTQC_cudaMemcpyAsync,
	VIRTQC_cudaMemset = 208, // no 207 since no data sent
	VIRTQC_cudaFree,

	// Device Management (runtime API)
	VIRTQC_cudaGetDevice, // 210
	VIRTQC_cudaGetDeviceCount = 212, 
	VIRTQC_cudaDeviceReset,
	VIRTQC_cudaSetDevice,
	VIRTQC_cudaDeviceSynchronize = 216, // 215 no args
	VIRTQC_cudaGetDeviceProperties,

	// Version Management (runtime API)
	VIRTQC_cudaDriverGetVersion,
	VIRTQC_cudaRuntimeGetVersion = 220,

	// Event Management (runtime API)
	VIRTQC_cudaEventCreate = 180,
	VIRTQC_cudaEventCreateWithFlags = 244, // 244
	VIRTQC_cudaEventRecord = 164, // 250
	VIRTQC_cudaEventSynchronize = 246, // 224
	VIRTQC_cudaEventElapsedTime = 248,
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
