#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <dlfcn.h>

#include <cuda_runtime.h>

#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>
#include <cuda.h>

#include "time_measure.h"
				
#if 0
#define pfunc() \
	print("### %s\n", __func__)
#else
#define pfunc() 
#endif

#if 0
#define ptrace(fmt, arg...) 	print("###    "fmt, ##arg)
#else
#define ptrace(fmt, arg...)
#endif

#define error() printf("###    ERROR: %s %d ##########################################\n", __func__, __LINE__);

void *lib = NULL;

CUmodule mod;
CUdevice dev;
CUcontext ctx;

CUfunction fun[8];
void* funPtr[8];
size_t funSize=0;

void *para[1024];
size_t paraSize=0;
dim3 gridDim;
dim3 blockDim;
size_t sharedMem;
cudaStream_t stream;

#define Errchk(ans) { DrvAssert((ans), __FILE__, __LINE__); }
inline void DrvAssert( CUresult code, const char *file, int line)
{
	char *str;
	if (code != CUDA_SUCCESS) {
		cuGetErrorName(code, (const char**)&str);
		printf("Error: %s at %s:%d\n", str, file, line);
		cuCtxDestroy(ctx);
		exit(code);
	}
}

void open_library()
{
	time_init();

	lib = dlopen("/usr/local/cuda/lib64/libcudart.so.7.5.18", RTLD_LAZY);
	if( !lib )
	{
		ptrace("open library failed, %s (%d)\n", strerror(errno), errno);
		exit (EXIT_FAILURE);
	}
}

void close_library()
{
	dlclose(lib);
	time_fini();
}

////////////////////////////////////////////////////////////////////////////////
///	implement by driver api
////////////////////////////////////////////////////////////////////////////////

void** __cudaRegisterFatBinary(void *fatCubin)
{
	unsigned int magic = *(unsigned int*)fatCubin;
	void **fatCubinHandle = malloc(sizeof(void*));

	time_init();
	open_library();

	pfunc();
	time_begin();
	ptrace("    fatCubin= %p\n", fatCubin);

	if( magic == FATBINC_MAGIC)
	{// fatBinaryCtl.h
		__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
		ptrace("    FATBINC_MAGIC\n");
		ptrace("    magic= %x\n", binary->magic);
		ptrace("    version= %x\n", binary->version);
		ptrace("    data= %p\n", binary->data);
		ptrace("    filename_or_fatbins= %p\n", binary->filename_or_fatbins);

		// TODO
		// add 0x50 is result of researching hexdump of binary file
		// it should be  some data struct element
		*fatCubinHandle = (void*)binary->data;// + 0x50;
	}
	else 
	{
#if 0	
magic: __cudaFatFormat.h
		   header: __cudaFatMAGIC)
		   __cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)fatCubin;

magic: FATBIN_MAGIC
		   header: fatbinary.h
		   computeFatBinaryFormat_t binary = (computeFatBinaryFormat_t)fatCubin;
#endif	 
	   ptrace("Unrecognized CUDA FAT MAGIC 0x%x\n", magic);
	   exit(1);
	}

	cuInit(0);
	cuDeviceGet(&dev, 0);
	cuCtxCreate(&ctx, 0, dev);

	// the pointer value is cubin ELF entry point
	time_end(t_RegFatbin);
	return fatCubinHandle;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{	
	pfunc();
	time_begin();

	ptrace("%s\n", __func__);
	ptrace("    handle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);

	cuCtxDestroy(ctx);
	free(fatCubinHandle);
	time_end(t_UnregFatbin);
	close_library();
}

void __cudaRegisterFunction(
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
	pfunc();
	time_begin();

	ptrace("    fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	ptrace("    hostFun= %s (%p)\n", hostFun, hostFun);
	ptrace("    deviceFun= %s (%p)\n", deviceFun, deviceFun);
	ptrace("    deviceName= %s\n", deviceName);
	ptrace("    thread_limit= %d\n", thread_limit);

	if(tid) ptrace("    tid= %u %u %u\n", tid->x, tid->y, tid->z);
	else	ptrace("    tid is NULL\n");

	if(bid)	ptrace("    bid= %u %u %u\n", bid->x, bid->y, bid->z);
	else	ptrace("    bid is NULL\n");

	if(bDim)ptrace("    bDim= %u %u %u\n", bDim->x, bDim->y, bDim->z);
	else	ptrace("    bDim is NULL\n");

	if(gDim)ptrace("    gDim= %u %u %u\n", gDim->x, gDim->y, gDim->z);
	else	ptrace("    gDim is NULL\n");

	if(wSize)ptrace("    wSize= %d\n", *wSize);
	else	 ptrace("    wSize is NULL\n");


	computeFatBinaryFormat_t fatBinHeader;
	unsigned long long int fatSize;
	char *fatBin;

	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);

	ptrace("    magic= %x\n", fatBinHeader->magic);
	ptrace("    version= %x\n", fatBinHeader->version);
	ptrace("    headerSize= %x\n", fatBinHeader->headerSize);
	ptrace("    fatSize= %llx\n", fatBinHeader->fatSize);

	fatSize = fatBinHeader->fatSize;
	fatBin = (char*)malloc(fatSize);
	memcpy(fatBin, fatBinHeader, fatSize);

	Errchk( cuModuleLoadData( &mod, fatBin ));
	Errchk( cuModuleGetFunction(&fun[funSize], mod, deviceName));
	funPtr[funSize] = (void*)hostFun;
	funSize++;

	free(fatBin);
	time_end(t_RegFunc);
}

cudaError_t cudaConfigureCall(
		dim3 _gridDim, 
		dim3 _blockDim, 
		size_t _sharedMem, 
		cudaStream_t _stream)
{
	pfunc();
	time_begin();

	ptrace("    gridDim= %d %d %d\n", _gridDim.x, _gridDim.y, _gridDim.z);
	ptrace("    blockDim= %d %d %d\n", _blockDim.x, _blockDim.y, _blockDim.z);
	ptrace("    sharedMem= %lu\n", _sharedMem);
	ptrace("    stream= %p\n", (void*)_stream);
	//ptrace("    size= %lu\n", sizeof(cudaStream_t));

	memcpy(  &gridDim,   &_gridDim, sizeof(dim3));
	memcpy( &blockDim,  &_blockDim, sizeof(dim3));
	memcpy(&sharedMem, &_sharedMem, sizeof(size_t));
	memcpy(   &stream,    &_stream, sizeof(cudaStream_t));

	time_end(t_ConfigCall);
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(
		const void *arg, 
		size_t size, 
		size_t offset)
{
	pfunc();
	time_begin();

	switch(size)
	{
		case 4:
			ptrace("    arg= %p, value= %u\n", arg, *(unsigned int*)arg);
			break;
		case 8:
			ptrace("    arg= %p, value= %llx\n", arg, *(unsigned long long*)arg);
			break;
	}

	ptrace("size= %lu\n", size);
	ptrace("offset= %lu\n", offset);

	/*
	   memcpy(para+offset, arg, size);
	   paraSize += size;
	 */

	para[ paraSize ] = (void*)arg;
	paraSize++;

	time_end(t_SetArg);
	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func)
{
	size_t funIdx;
	pfunc();
	time_begin();

	ptrace("    func= %p\n", func);
	ptrace("    gridDim= %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
	ptrace("    blockDim= %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
	ptrace("    sharedMem= %lu\n", sharedMem);
	ptrace("    stream= %p\n", stream);

	for(funIdx=0; funIdx<funSize; funIdx++)
		if(funPtr[funIdx]==func) break;

	ptrace("funSize= %lu, idx= %lu, fun= %p\n", 
			funSize, funIdx, (void*)fun[funIdx]);

	Errchk(cuLaunchKernel(fun[funIdx], 
				gridDim.x,  gridDim.y,  gridDim.z,
				blockDim.x, blockDim.y, blockDim.z, 
				sharedMem, stream, para, NULL));
	paraSize = 0;

	time_end(t_Launch);
	return cudaSuccess;
}

////////////////////////////////////////////////////////////////////////////////
/// direct call runtime api
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaGetDevice(int *device)
{
	cudaError_t (*func_L)(int*);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaGetDevice");
	err = (*func_L)(device);

	if( err != cudaSuccess ) error();
	time_end(t_GetDev);
	return err;
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	cudaError_t (*func)(void**, size_t);
	cudaError_t err;
	pfunc();
	time_begin();

	ptrace("devPtr= %p, size= %lu\n", *devPtr, size);
	func = dlsym(lib, "cudaMalloc");
	err = (*func)(devPtr, size);
	ptrace("devPtr= %p, size= %lu\n", *devPtr, size);

	if( err != cudaSuccess ) error();
	time_end(t_Malloc);
	return err;
}

cudaError_t cudaFree(void* devPtr)
{
	cudaError_t (*func)(void*);
	cudaError_t err;
	pfunc();
	time_begin();

	ptrace("devPtr= %p\n", devPtr);
	func = dlsym(lib, "cudaFree");
	err = (*func)(devPtr);

	if( err != cudaSuccess ) error();
	time_end(t_Free);
	return err;
}
/*
cudaError_t cudaFreeHost(void *ptr) 	
{
	cudaError_t (*func)(void*);
	cudaError_t err;

	pfunc();
	time_begin();

	func = dlsym(lib, "cudaFreeHost");
	err = (*func)(ptr);

	if( err != cudaSuccess ) error();

	time_end();
	return err;
}
*/
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,  enum cudaMemcpyKind kind)
{
	cudaError_t (*func)(void*, const void*, size_t, enum cudaMemcpyKind);
	cudaError_t err;
	pfunc();
	time_begin();

	ptrace("dst= %p, src= %p, count= %lu, kind %d\n", dst, src, count, kind);
	func = dlsym(lib, "cudaMemcpy");
	err = (*func)(dst, src, count, kind);

	if( err != cudaSuccess ) error();

	if(kind==1){
		time_end(t_MemcpyH2D);
	}else if(kind==2){
		time_end(t_MemcpyD2H);
	}
	return err;
}


cudaError_t cudaGetDeviceCount(int *count) 	
{
	cudaError_t (*func_L)(int*);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaGetDeviceCount");
	err = (*func_L)(count);

	if( err != cudaSuccess ) error();
	time_end(t_GetDevCount);
	return err;
}


cudaError_t cudaSetDevice(int device) 	
{
	cudaError_t (*func_L)(int);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaSetDevice");
	err = (*func_L)(device);

	if( err != cudaSuccess ) error();

	time_end(t_SetDev);
	return err;
}

cudaError_t cudaDriverGetVersion(int *driverVersion)
{
	cudaError_t (*func_L)(int*);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaDriverGetVersion");
	err = (*func_L)(driverVersion);

	if( err != cudaSuccess ) error();
	time_end(t_DriverGetVersion);
	return err;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	cudaError_t (*func_L)(int*);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaRuntimeGetVersion");
	err = (*func_L)(runtimeVersion);

	if( err != cudaSuccess ) error();
	time_end(t_RuntimeGetVersion);
	return err;
}


cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	cudaError_t (*func_L)(struct cudaDeviceProp *, int);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaGetDeviceProperties");
	err = (*func_L)(prop, device);

	if( err != cudaSuccess ) error();
	time_end(t_GetDevProp);
	return err;
}

cudaError_t cudaDeviceSynchronize(void) 	
{
	cudaError_t (*func_L)(void);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaDeviceSynchronize");
	err = (*func_L)();

	if( err != cudaSuccess ) error();
	time_end(t_DevSync);
	return err;
}

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
	cudaError_t (*func_L)(cudaEvent_t *);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaEventCreate");
	err = (*func_L)(event);

	if( err != cudaSuccess ) error();
	time_end(t_EventCreate);
	return err;
}

cudaError_t cudaEventRecord(cudaEvent_t event,	cudaStream_t stream)
{
	cudaError_t (*func_L)(cudaEvent_t, cudaStream_t);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaEventRecord");
	err = (*func_L)(event, stream);

	if( err != cudaSuccess ) error();
	time_end(t_EventRecord);
	return err;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) 
{
	cudaError_t (*func_L)(cudaEvent_t);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaEventSynchronize");
	err = (*func_L)(event);

	if( err != cudaSuccess ) error();
	time_end(t_EventSync);
	return err;
}


cudaError_t cudaEventElapsedTime(float *ms,	cudaEvent_t start, cudaEvent_t end)	
{
	cudaError_t (*func_L)(float*, cudaEvent_t, cudaEvent_t);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaEventElapsedTime");
	err = (*func_L)(ms, start, end);

	if( err != cudaSuccess ) error();
	time_end(t_EventElapsedTime);
	return err;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) 	
{
	cudaError_t (*func_L)(cudaEvent_t);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaEventDestroy");
	err = (*func_L)(event);

	if( err != cudaSuccess ) error();
	time_end(t_EventDestroy);
	return err;
}

cudaError_t cudaDeviceReset(void)
{
	cudaError_t (*func_L)(void);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaDeviceReset");
	err = (*func_L)();

	if( err != cudaSuccess ) error();
	time_end(t_DevReset);
	return err;
}

cudaError_t cudaGetLastError(void)
{
	cudaError_t (*func_L)(void);
	cudaError_t err;
	pfunc();
	time_begin();

	func_L = dlsym(lib, "cudaGetLastError");
	err = (*func_L)();

	if( err != cudaSuccess ) error();
	time_end(t_GetLastError);
	return err;
}


const char* cudaGetErrorString(cudaError_t error)
{
	const char* (*func_L)(cudaError_t);
	char* str;
	time_begin();

	func_L = dlsym(lib, "cudaGetErrorString");
	str = (char*)(*func_L)(error);

	return (const char*)str;
}
