// note using _CRT_SECURE_NO_WARNINGS
// when compiling to avoid error
// this should be fix and it is not thread safe and has security issues
// strerror should be replaced with strerror_s 
// different signature

#include <windows.h>
#include <winioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h> // open
#include <sddl.h>
#include <aclapi.h>
#include <malloc.h> // unaligned_malloc/free

/*
#include <unistd.h> // close
#include <sys/ioctl.h> // ioclt
#include <sys/mman.h>
*/

#include <builtin_types.h>
#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>

#define DllExport __declspec( dllexport )
#define DllImport __declspec( dllimport )
#define malloc(x) qcu_malloc(x)
#define free(x) qcu_free(x);

//DllExport void * qcu_malloc(uint64_t size);
//DllExport void qcu_free(void* tptr);

#include "time_measure.h"
#include "../qcuda/qcuda_common.h"

//#define PAGE_SIZE 4096 //hardcode work
#define _4KB 4096
#define _1MB (1024 * 1024)
#define _2MB (2 * _1MB)
#define _4MB (2 * _2MB)
#define Q_PAGE_SIZE _4KB

#if 0
#define pfunc() printf("### %s at line %d\n", __FUNCTION__, __LINE__)
#else
#define pfunc()
#endif

#if 0
#define ptrace(fmt, ...) \
	printf("    " fmt, ##__VA_ARGS__)
#else
#define ptrace(fmt, ...)
#endif

#define error(fmt, ...) printf("ERROR: " fmt, ##__VA_ARGS__)

#define ptr( p , v, s) \
	p = (uint64_t)v; \
	p##Size = (uint32_t)s;

HANDLE fd;
unsigned int map_offset = 0;

extern void *_aligned_malloc(size_t, size_t);
extern void _aligned_free(void*);

//uint32_t cudaKernelConf[7];
uint64_t cudaKernelConf[8];

#define cudaKernelParaMaxSize 128
uint8_t cudaKernelPara[cudaKernelParaMaxSize];
uint32_t cudaParaSize;
SECURITY_ATTRIBUTES *pSec = 0;

HANDLE hMapFile;


//void *myfat = (int[]){0}; //cocotion

/////////////////////////////////////////////////////////////////////////////////
// Import functions
/////////////////////////////////////////////////////////////////////////////////
// remove advapi32.dll dependency :( silly visual studio
BOOLEAN WINAPI SystemFunction036(PVOID RandomBuffer, ULONG RandomBufferLength)
{
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// General Function
////////////////////////////////////////////////////////////////////////////////

void DisplayError(TCHAR* pszAPI, DWORD dwError)
{
	LPVOID lpvMessageBuffer;

	FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, dwError,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPTSTR)&lpvMessageBuffer, 0, NULL);

	//... now display this string
	printf(TEXT("ERROR: API        = %s\n"), pszAPI);
	printf(TEXT("       error code = %d\n"), dwError);
	printf(TEXT("       message    = %s\n"), lpvMessageBuffer);

	// Free the buffer allocated by the system
	LocalFree(lpvMessageBuffer);
	getch();
	ExitProcess(GetLastError());
}

void Privilege(TCHAR* pszPrivilege, BOOL bEnable)
{
	HANDLE           hToken;
	TOKEN_PRIVILEGES tp;
	BOOL             status;
	DWORD            error;

	// open process token
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
		DisplayError(TEXT("OpenProcessToken"), GetLastError());

	// get the luid
	if (!LookupPrivilegeValue(NULL, pszPrivilege, &tp.Privileges[0].Luid))
		DisplayError(TEXT("LookupPrivilegeValue"), GetLastError());

	tp.PrivilegeCount = 1;

	// enable or disable privilege
	if (bEnable)
		tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
	else
		tp.Privileges[0].Attributes = 0;

	// enable or disable privilege
	status = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);

	// It is possible for AdjustTokenPrivileges to return TRUE and still not succeed.
	// So always check for the last error value.
	error = GetLastError();
	if (!status || (error != ERROR_SUCCESS))
		DisplayError(TEXT("AdjustTokenPrivileges"), GetLastError());

	// close the handle
	if (!CloseHandle(hToken))
		DisplayError(TEXT("CloseHandle"), GetLastError());
}

void open_device()
{
	pfunc();
	/*SECURITY_ATTRIBUTES security;
	ZeroMemory(&security, sizeof(security));
	security.nLength = sizeof(security);

	ConvertStringSecurityDescriptorToSecurityDescriptor(
		L"D:P(A;;GA;;;SY)(A;;GA;;;BA)",
		SDDL_REVISION_1,
		&security.lpSecurityDescriptor,
		NULL);*/
	////
	//SecAttr.nLength = sizeof(SecAttr);
	//SecAttr.lpSecurityDescriptor = &SecDesc;
	//SecAttr.bInheritHandle = FALSE;

	//InitializeSecurityDescriptor(SecAttr.lpSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);
	//SetSecurityDescriptorDacl(pSD, TRUE, (PACL)0, FALSE);
	////
	//GetSecurityDescriptorSacl(pSD, &fSaclPresent, &pSacl, &fSaclDefaulted);
	//SetSecurityDescriptorSacl(SecAttr.lpSecurityDescriptor, TRUE, pSacl, FALSE);
	// pSec = &security;

	LPCWSTR dev_path = L"\\\\.\\qcuda"; // \\qcuda.txt
	fd = CreateFile(dev_path, (FILE_GENERIC_READ | FILE_GENERIC_WRITE),
		(FILE_SHARE_READ | FILE_SHARE_WRITE), // FILE_SHARE_DELETE
		NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL); // FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED

	if (fd == INVALID_HANDLE_VALUE)
	{
		error("open device %ls failed, %s (%u)\n", dev_path, strerror(errno), GetLastError());
		exit(EXIT_FAILURE);
	}

	/*SetSecurityInfo(fd, SE_KERNEL_OBJECT,
	DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
	NULL, NULL, NULL, NULL);*/
	//printf("open is ok fd: %d\n", fd);
}

void send_cmd_to_device(int cmd, VirtioQCArg *arg, int submitType)
{
	size_t len = (arg) ? sizeof(VirtioQCArg) : 0;
	ULONG returnedControl;
	/*DWORD transferredBytes;
	BOOL fOverlapped;

	DeviceIoOverlapped.Offset = 0;
	DeviceIoOverlapped.OffsetHigh = 0;
	DeviceIoOverlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);*/

	ptrace("about to send ioctl command %d submitType=%d, len=%d\n", cmd, submitType, (int)len);
	returnedControl = DeviceIoControl(fd, // device handler
		cmd, // command to send
		(submitType == 1 || submitType == 3) ? arg : NULL, // command arguments
		(submitType == 1 || submitType == 3) ? len : 0, //
		(submitType == 2 || submitType == 3) ? arg : NULL, // response
		(submitType == 2 || submitType == 3) ? len : 0, // response size
		NULL, // size of data in output buffer
		NULL); // (LPOVERLAPPED) overlapped structure
}

void close_device()
{
	pfunc();
	CloseHandle(fd);
	time_fini();
}

void *__zcmalloc(uint64_t size)
{
	pfunc();
	/*DWORD fileSize = GetFileSize(fd, &fileSize);
	SYSTEM_INFO si;
	GetSystemInfo(&si);*/
	void *addr = NULL;
	VirtioQCArg arg;
	//	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
	int blocksize = Q_PAGE_SIZE; // si.dwPageSize * 1024; //test 4096k
	unsigned int numOfBlocks = size / blocksize;
	if (size%blocksize != 0) numOfBlocks++;

	unsigned int bytesAllocated = numOfBlocks*blocksize;

	ptrace("blocksize=%d, size=%d, numOfblocks=%d\n\n", blocksize, size, numOfBlocks);
	
	Privilege(TEXT("SeLockMemoryPrivilege"), TRUE);

	addr = VirtualAlloc(NULL, 
		bytesAllocated,
		MEM_COMMIT | MEM_RESERVE, //  | Q_PAGE_SIZE, //MEM_LARGE_PAGES, 
		PAGE_READWRITE);

	if (!addr)
	{
		printf("allocation of %d Bytes failed\n", bytesAllocated);
		exit(-1);
	}

	/*if (VirtualLock(addr, bytesAllocated))
	{
		printf("locking address of size %d Bytes failed\n", bytesAllocated);
		VirtualFree(addr, bytesAllocated, MEM_RELEASE);
		exit(-1);
	}*/
	
	ptr(arg.pA, addr, bytesAllocated);
	send_cmd_to_device(VIRTQC_CMD_MMAP, &arg, 3);

	// map_offset = numOfBlocks*blocksize;
	
	//memset(&arg, 0, sizeof(VirtioQCArg));
	//ptr(arg.pA, size, sizeof(size));
	//// ptr(arg.pB, numOfBlocks, sizeof(numOfBlocks));
	//send_cmd_to_device(VIRTQC_CMD_MMAP, &arg, 3);
	//if (arg.flag == 1)
	//	addr = (void*)arg.pA;
	//
	return addr;
}

DllExport void * qcu_malloc(uint64_t size)
{
	pfunc();
#ifdef USER_KERNEL_COPY
	if (size > QCU_KMALLOC_MAX_SIZE)
#endif
		return __zcmalloc(size);
#ifdef USER_KERNEL_COPY
	else
#if _WIN64
		return _aligned_malloc(size, 16);
#elif _WIN32
		return _aligned_malloc(size, 8);
#endif
#endif
	ptrace("end of qcu_malloc\n");
}

DllExport void qcu_free(void* tptr)
{
	pfunc();
	VirtioQCArg arg;

	ptr(arg.pA, tptr, 0);

	send_cmd_to_device(VIRTQC_CMD_MMAPRELEASE, &arg, 3);

	if (arg.flag == 1) {
		//VirtualUnlock(arg.pA, arg.pASize);
		VirtualFree(arg.pA, arg.pASize, MEM_RELEASE);
	}

#ifdef USER_KERNEL_COPY
	if ((int)arg.cmd == -1)
		_aligned_free(ptr);
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// Module & Execution control
////////////////////////////////////////////////////////////////////////////////

DllExport void** __cudaRegisterFatBinary(void *fatCubin)
{
	unsigned int magic;
	void **fatCubinHandle;
	time_init();
	pfunc();
	open_device();
	time_begin();

	magic = *(unsigned int*)fatCubin;
	fatCubinHandle = malloc(sizeof(void*)); //original
											//	fatCubinHandle = myfat; //cocotion

	if (magic == FATBINC_MAGIC)
	{// fatBinaryCtl.h
		__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
		ptrace("FATBINC_MAGIC\n");
		ptrace("magic= %x\n", binary->magic);
		ptrace("version= %x\n", binary->version);
		ptrace("data= %p\n", binary->data);
		ptrace("filename_or_fatbins= %p\n", binary->filename_or_fatbins);

		*fatCubinHandle = (void*)binary->data;
	}
	else
	{
		/*
		magic: __cudaFatFormat.h
		header: __cudaFatMAGIC)
		__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)fatCubin;

		magic: FATBIN_MAGIC
		header: fatbinary.h
		computeFatBinaryFormat_t binary = (computeFatBinaryFormat_t)fatCubin;
		*/
		ptrace("Unrecognized CUDA FAT MAGIC 0x%x\n", magic);
		exit(EXIT_FAILURE);
	}

	send_cmd_to_device(VIRTQC_cudaRegisterFatBinary, NULL, 0);

	// the pointer value is cubin ELF entry point
	time_end(t_RegFatbin);

	return fatCubinHandle;
}

DllExport void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	pfunc();
	time_begin();
	ptrace("fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	send_cmd_to_device(VIRTQC_cudaUnregisterFatBinary, NULL, 0);

	free(fatCubinHandle);
	
	time_end(t_UnregFatbin);
	close_device();
}

DllExport void __cudaRegisterFunction(
	void   **fatCubinHandle,
	const char    *hostFun,
	char    *deviceFun,
	const char    *deviceName,
	int      thread_limit,
	uint3   *tid,
	uint3   *bid,
	dim3    *bDim,
	dim3    *gDim,
	int     *wSize
	)
{
	VirtioQCArg arg;
	computeFatBinaryFormat_t fatBinHeader;
	pfunc();
	time_begin();
	
	ptrace("fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	ptrace("hostFun= %s (%p)\n", hostFun, hostFun);
	ptrace("deviceFun= %s (%p)\n", deviceFun, deviceFun);
	ptrace("deviceName= %s\n", deviceName);
	ptrace("thread_limit= %d\n", thread_limit);

	if (tid) ptrace("tid= %u %u %u\n", tid->x, tid->y, tid->z);
	else	ptrace("tid is NULL\n");

	if (bid)	ptrace("bid= %u %u %u\n", bid->x, bid->y, bid->z);
	else	ptrace("bid is NULL\n");

	if (bDim)ptrace("bDim= %u %u %u\n", bDim->x, bDim->y, bDim->z);
	else	ptrace("bDim is NULL\n");

	if (gDim)ptrace("gDim= %u %u %u\n", gDim->x, gDim->y, gDim->z);
	else	ptrace("gDim is NULL\n");

	if (wSize)ptrace("wSize= %d\n", *wSize);
	else	 ptrace("wSize is NULL\n");

	memset(&arg, 0, sizeof(VirtioQCArg));
	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);

	ptr(arg.pA, fatBinHeader, fatBinHeader->fatSize);
	ptr(arg.pB, deviceName, strlen(deviceName) + 1);
	arg.flag = (uint32_t)(uint64_t)hostFun;

	ptrace("pA= %p, pASize= %u, pB= %p, pBSize= %u\n",
		(void*)arg.pA, arg.pASize, (void*)arg.pB, arg.pBSize);

	send_cmd_to_device(VIRTQC_cudaRegisterFunction, &arg, 3);
	
	time_end(t_RegFunc);
}

DllExport cudaError_t cudaConfigureCall(
	dim3 gridDim,
	dim3 blockDim,
	size_t sharedMem,
	cudaStream_t stream)
{
	pfunc();
	time_begin();

	ptrace("gridDim= %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
	ptrace("blockDim= %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
	ptrace("sharedMem= %lu\n", sharedMem);
	ptrace("stream= %p\n", (void*)stream);
	//ptrace("size= %lu\n", sizeof(cudaStream_t));

	cudaKernelConf[0] = gridDim.x;
	cudaKernelConf[1] = gridDim.y;
	cudaKernelConf[2] = gridDim.z;

	cudaKernelConf[3] = blockDim.x;
	cudaKernelConf[4] = blockDim.y;
	cudaKernelConf[5] = blockDim.z;

	cudaKernelConf[6] = sharedMem;

	cudaKernelConf[7] = (stream == NULL) ? (uint64_t)-1 : (uint64_t)stream;


	memset(cudaKernelPara, 0, cudaKernelParaMaxSize);
	cudaParaSize = sizeof(uint32_t);

	time_end(t_ConfigCall);
	return cudaSuccess;
}

DllExport cudaError_t cudaSetupArgument(
	const void *arg,
	size_t size,
	size_t offset)
{
	pfunc();
	time_begin();
	/*
	cudaKernelPara:
	uint32_t      uint32_t                   uint32_t
	=============================================================================
	| number of arg | arg1 size |  arg1 data  |  arg2 size  |  arg2 data  | .....
	=============================================================================
	*/
	// set data size
	memcpy(&cudaKernelPara[cudaParaSize], &size, sizeof(uint32_t));
	ptrace("cudaParaSize = %u, size= %u, cudaKernelPara[cudaParaSize]= %u\n",
		cudaParaSize, size, *(uint32_t*)&cudaKernelPara[cudaParaSize]);
	cudaParaSize += sizeof(uint32_t);

	// set data
	memcpy(&cudaKernelPara[cudaParaSize], arg, size);
	ptrace("value= %llx\n", *(unsigned long long*)&cudaKernelPara[cudaParaSize]);
	cudaParaSize += size;
	ptrace("cudaParaSize = %u, size= %u\n", cudaParaSize, size);

	(*((uint32_t*)cudaKernelPara))++;

	time_end(t_SetArg);
	return cudaSuccess;
}

DllExport cudaError_t cudaLaunch(const void *func)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();
	
	memset(&arg, 0, sizeof(VirtioQCArg));
	//	ptr( arg.pA, cudaKernelConf, 7*sizeof(uint32_t));
	ptr(arg.pA, cudaKernelConf, 8 * sizeof(uint64_t));
	ptr(arg.pB, cudaKernelPara, cudaParaSize);
	arg.flag = (uint32_t)(uint64_t)func;
	
	ptrace("devPtr= %p, arg.pB=%p, arg.flag=%lld\n", (void*)arg.pA, (void*)arg.pB, (long long int)arg.flag);

	send_cmd_to_device(VIRTQC_cudaLaunch, &arg, 3);
	
	time_end(t_Launch);
	return cudaSuccess;
}

////////////////////////////////////////////////////////////////////////////////
/// Memory Management
////////////////////////////////////////////////////////////////////////////////

DllExport cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr(arg.pA, 0, 0);
	arg.flag = size;

	send_cmd_to_device(VIRTQC_cudaMalloc, &arg, 3);
	*devPtr = (void*)arg.pA;
	ptrace("devPtr= %p\n", (void*)arg.pA);

	time_end(t_Malloc);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
	VirtioQCArg arg;

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr(arg.pA, devPtr, count);
	arg.para = value;

	send_cmd_to_device(VIRTQC_cudaMemset, &arg, 3);

	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaFree(void* devPtr)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr(arg.pA, devPtr, 0);

	send_cmd_to_device(VIRTQC_cudaFree, &arg, 3);
	ptrace("devPtr= %p\n", (void*)arg.pA);

	time_end(t_Free);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaMemcpy(
	void* dst,
	const void* src,
	size_t count,
enum cudaMemcpyKind kind)
{
	//struct timeval start, stop, diff;
	//gettimeofday(&start, NULL);
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptrace("dst= %p , src= %p ,size= %lu\n", (void*)dst, (void*)src, count);

	if (kind == cudaMemcpyHostToDevice)
	{
		ptr(arg.pA, dst, 0);
		ptr(arg.pB, src, count);
		arg.flag = 1;
		
		//gettimeofday(&stop, NULL);
		//diff.tv_sec = stop.tv_sec - start.tv_sec;
		//diff.tv_usec = stop.tv_usec - start.tv_usec;

		//printf("H2D = %lu sec, %lu usec\n", diff.tv_sec, diff.tv_usec);
	}
	else if (kind == cudaMemcpyDeviceToHost)
	{
		ptr(arg.pA, dst, count);
		ptr(arg.pB, src, 0);
		arg.flag = 2;

		//gettimeofday(&stop, NULL);
		//diff.tv_sec = stop.tv_sec - start.tv_sec;
		//diff.tv_usec = stop.tv_usec - start.tv_usec;

		//printf("H2D = %lu sec, %lu usec\n", diff.tv_sec, diff.tv_usec);
	}
	else if (kind == cudaMemcpyDeviceToDevice)
	{
		ptr(arg.pA, dst, 0);
		ptr(arg.pB, src, count);
		arg.flag = 3;
	}
	else
	{
		error("Not impletment cudaMemcpyKind %d\n", kind);
		return cudaErrorInvalidValue;
	}

	send_cmd_to_device(VIRTQC_cudaMemcpy, &arg, 3);

	if (kind == 1) {
		time_end(t_MemcpyH2D);
	}
	else if (kind == 2) {
		time_end(t_MemcpyD2H);
	}

	//gettimeofday(&stop, NULL);
	//diff.tv_sec = stop.tv_sec - start.tv_sec;
	//diff.tv_usec = stop.tv_usec - start.tv_usec;

	//printf("H2D = %lu sec, %lu usec\n", diff.tv_sec, diff.tv_usec);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaMemcpyAsync(
	void * 	dst,
	const void * 	src,
	size_t 	count,
enum cudaMemcpyKind 	kind,
	cudaStream_t 	stream)
{
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));

	uint64_t mystream = (stream == NULL) ? (uint64_t)-1 : (uint64_t)stream;

	if (kind == cudaMemcpyHostToDevice)
	{
		ptr(arg.pA, dst, 0);
		ptr(arg.pB, src, count);
		//arg.rnd = (uint64_t)stream;
		arg.rnd = mystream;
		arg.flag = 1;

		//gettimeofday(&stop, NULL);
		//diff.tv_sec = stop.tv_sec - start.tv_sec;
		//diff.tv_usec = stop.tv_usec - start.tv_usec;

		//printf("H2D = %lu sec, %lu usec\n", diff.tv_sec, diff.tv_usec);
	}
	else if (kind == cudaMemcpyDeviceToHost)
	{
		ptr(arg.pA, dst, count);
		ptr(arg.pB, src, 0);
		//arg.rnd = (uint64_t)stream;
		arg.rnd = mystream;
		arg.flag = 2;

		//gettimeofday(&stop, NULL);
		//diff.tv_sec = stop.tv_sec - start.tv_sec;
		//diff.tv_usec = stop.tv_usec - start.tv_usec;

		//printf("H2D = %lu sec, %lu usec\n", diff.tv_sec, diff.tv_usec);
	}
	else if (kind == cudaMemcpyDeviceToDevice)
	{
		ptr(arg.pA, dst, 0);
		ptr(arg.pB, src, count);
		//arg.rnd = (uint64_t)stream;
		arg.rnd = mystream;
		arg.flag = 3;
	}
	else
	{
		error("Not impletment cudaMemcpyKind %d\n", kind);
		return cudaErrorInvalidValue;
	}

	send_cmd_to_device(VIRTQC_cudaMemcpyAsync, &arg, 3);
	//gettimeofday(&stop, NULL);
	//diff.tv_sec = stop.tv_sec - start.tv_sec;
	//diff.tv_usec = stop.tv_usec - start.tv_usec;

	//printf("H2D = %lu sec, %lu usec\n", diff.tv_sec, diff.tv_usec);
	return (cudaError_t)arg.cmd;
}


////////////////////////////////////////////////////////////////////////////////
/// Device Management
////////////////////////////////////////////////////////////////////////////////

DllExport cudaError_t cudaGetDevice(int *device)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaGetDevice, &arg, 3);
	*device = (int)arg.pA;

	time_end(t_GetDev);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaGetDeviceCount(int *count)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaGetDeviceCount, &arg, 3);
	*count = (int)arg.pA;

	time_end(t_GetDevCount);
	return (cudaError_t)arg.cmd;
}
//////////////////////
/*cudaError_t checkCudaCapabilities(int m, int s)
{
VirtioQCArg arg;
memset(&arg, 0, sizeof(VirtioQCArg));

ptr( arg.pA, m, 0);
ptr( arg.pB, s, 0);

send_cmd_to_device( VIRTQC_checkCudaCapabilities, &arg);

return (cudaError_t)arg.cmd;
}
*/
/////////////////////
DllExport cudaError_t cudaSetDevice(int device)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, device, 0);
	send_cmd_to_device(VIRTQC_cudaSetDevice, &arg, 3);

	time_end(t_SetDev);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, prop, sizeof(struct cudaDeviceProp));
	ptr(arg.pB, device, 0);
	send_cmd_to_device(VIRTQC_cudaGetDeviceProperties, &arg, 3);

	time_end(t_GetDevProp);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaDeviceSynchronize(void)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaDeviceSynchronize, &arg, 3);

	time_end(t_DevSync);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaDeviceReset(void)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaDeviceReset, &arg, 3);

	time_end(t_DevReset);
	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Version Management
////////////////////////////////////////////////////////////////////////////////

DllExport cudaError_t cudaDriverGetVersion(int *driverVersion)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaDriverGetVersion, &arg, 3);
	*driverVersion = (int)arg.pA;

	time_end(t_DriverGetVersion);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaRuntimeGetVersion, &arg, 3);
	*runtimeVersion = (uint64_t)arg.pA;

	time_end(t_RuntimeGetVersion);
	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Event Management
////////////////////////////////////////////////////////////////////////////////

DllExport cudaError_t cudaEventCreate(cudaEvent_t *event)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaEventCreate, &arg, 3);

	*event = (void*)arg.pA;
	ptrace("cudaEventCreate--->>%lld\n", (long long int)arg.pA);

	time_end(t_EventCreate);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
	VirtioQCArg arg;

	memset(&arg, 0, sizeof(VirtioQCArg));

	arg.flag = flags;
	ptrace("eventCreateWithFlags--->%lld",(long long int)arg.flag);
	send_cmd_to_device(VIRTQC_cudaEventCreateWithFlags, &arg, 3);

	*event = (void*)arg.pA;
	ptrace("cudaWithFlags Result--->>%lld\n", (long long int)arg.pA);

	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	uint64_t mystream = (stream == NULL) ? (uint64_t)-1 : (uint64_t)stream;

	ptr(arg.pA, event, 0);
	//ptr( arg.pB, stream, 0);
	ptr(arg.pB, mystream, 0);
	ptrace("cudaEventRecord--->>%lld,%lld\n", (long long int)arg.pA, (long long int)arg.pB);
	send_cmd_to_device(VIRTQC_cudaEventRecord, &arg, 3);

	time_end(t_EventRecord);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, event, 0);
	send_cmd_to_device(VIRTQC_cudaEventSynchronize, &arg, 3);

	time_end(t_EventSync);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, start, 0);
	ptr(arg.pB, end, 0);
	send_cmd_to_device(VIRTQC_cudaEventElapsedTime, &arg, 3);

	memcpy(ms, &arg.flag, sizeof(float));
	ptrace("----------------------->elapsed time:: %lld\n", (long long int)arg.flag);

	time_end(t_EventElapsedTime);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, event, 0);
	send_cmd_to_device(VIRTQC_cudaEventDestroy, &arg, 3);

	time_end(t_EventDestroy);
	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Error Handling
////////////////////////////////////////////////////////////////////////////////

DllExport cudaError_t cudaGetLastError(void)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device(VIRTQC_cudaGetLastError, &arg, 3);

	time_end(t_GetLastError);
	return (cudaError_t)arg.cmd;
}

DllExport const char* cudaGetErrorString(cudaError_t 	error)
{
	return "Not yet implement";
}

////////about zero-copy

DllExport cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr(arg.pA, ptr, size);
	arg.flag = flags;

	send_cmd_to_device(VIRTQC_cudaHostRegister, &arg, 3);

	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaHostGetDevicePointer(void ** pDevice, void *pHost, unsigned int flags)
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, 0, 0);
	ptr(arg.pB, pHost, 0);
	arg.flag = flags;

	send_cmd_to_device(VIRTQC_cudaHostGetDevicePointer, &arg, 3);
	*pDevice = (void*)arg.pA;

	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaHostUnregister(void *ptr)
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, ptr, 0);

	send_cmd_to_device(VIRTQC_cudaHostUnregister, &arg, 3);

	return (cudaError_t)arg.cmd;
}

/////////////////////////////////////
//about stream
/////////////////////////////////////
DllExport cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));
	//ptr( arg.pA, pStream,  0);
	send_cmd_to_device(VIRTQC_cudaStreamCreate, &arg, 3);

	*pStream = (cudaStream_t)arg.pA;
	ptrace("cudaStreamCreate--->>%lld\n", (long long int)arg.pA);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	VirtioQCArg arg;

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, stream, 0);
	send_cmd_to_device(VIRTQC_cudaStreamDestroy, &arg, 3);

	return (cudaError_t)arg.cmd;
}

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

DllExport cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	void *a_UA = malloc(size + MEMORY_ALIGNMENT);
	*pHost = (void *)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);

	return cudaHostRegister(*pHost, size, flags);
}

DllExport cudaError_t cudaMallocHost(void **pHost, size_t size)
{
	return cudaHostAlloc(pHost, size, cudaHostAllocDefault);
}

DllExport cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));
	arg.flag = flags;
	ptrace("cudaSetDeviceFlags---->> %lld\n", (long long int)arg.flag);
	send_cmd_to_device(VIRTQC_cudaSetDeviceFlags, &arg, 3);
	return (cudaError_t)arg.cmd;
}

DllExport cudaError_t cudaFreeHost(void *ptr)
{
	cudaHostUnregister(ptr);
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr(arg.pA, ptr, 0);
	send_cmd_to_device(VIRTQC_cudaFreeHost, &arg, 3);

	if (arg.flag == 1)
	{
		//VirtualUnlock(arg.pA, arg.pASize);
		VirtualFree(arg.pA, arg.pASize, MEM_RELEASE);
	}

	return (cudaError_t)arg.cmd;
}