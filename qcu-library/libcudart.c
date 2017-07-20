#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h> // open
#include <unistd.h> // close
#include <sys/ioctl.h> // ioclt
#include <sys/mman.h>

#include <builtin_types.h>
#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>

#include "time_measure.h"
#include "../qcu-driver/qcuda_common.h"

#if 0
#define pfunc() printf("### %s at line %d\n", __func__, __LINE__)
#else
#define pfunc()
#endif

#if 0
#define ptrace(fmt, arg...) \
	printf("    " fmt, ##arg)
#else
#define ptrace(fmt, arg...)
#endif

#define dev_path "/dev/qcuda"

#define error(fmt, arg...) printf("ERROR: "fmt, ##arg)

#define ptr( p , v, s) \
	p = (uint64_t)v; \
	p##Size = (uint32_t)s;

int fd = -1;
unsigned int map_offset = 0;

extern void *__libc_malloc(size_t);
extern void __libc_free(void*);

//uint32_t cudaKernelConf[7];
uint64_t cudaKernelConf[8];

#define cudaKernelParaMaxSize 128
uint8_t cudaKernelPara[ cudaKernelParaMaxSize ];
uint32_t cudaParaSize;


////////////////////////////////////////////////////////////////////////////////
/// General Function
////////////////////////////////////////////////////////////////////////////////

void open_device()
{
	fd = open(dev_path, O_RDWR);
	if( fd < 0 )
	{
		error("open device %s faild, %s (%d)\n", dev_path, strerror(errno), errno);
		exit (EXIT_FAILURE);
	}
	//printf("open is ok fd: %d\n", fd);
}

void close_device()
{
	close(fd);
	time_fini();
}

void send_cmd_to_device(int cmd, VirtioQCArg *arg)
{
//	if(__builtin_expect(!!(fd==-1), 0))
//		open_device();

	ioctl(fd, cmd, arg);
}

void *__zcmalloc(uint64_t size)
{
	time_begin();

	int blocksize = getpagesize()*1024; //test 4096k	
	unsigned int numOfblocks = size/blocksize;
	if(size%blocksize != 0) numOfblocks++;

	void *addr = mmap(0, numOfblocks*blocksize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map_offset);
		
	msync(addr, numOfblocks*blocksize, MS_ASYNC);
	
	map_offset+=numOfblocks*blocksize;
	
	time_end(t_myMalloc);
	
	return addr;
}

void *malloc(uint64_t size)
{
#ifdef USER_KERNEL_COPY		
	if( size > QCU_KMALLOC_MAX_SIZE)
#endif
		return __zcmalloc(size);
#ifdef USER_KERNEL_COPY		
	else
		return __libc_malloc(size);
#endif
}

void free(void* ptr)
{
	VirtioQCArg arg;
	ptr(arg.pA, ptr, 0);
	send_cmd_to_device(VIRTQC_CMD_MMAPRELEASE, &arg);

#ifdef USER_KERNEL_COPY		
	if((int)arg.cmd == -1)
		__libc_free(ptr);
#endif
}


//void send_cmd_to_device(int cmd, VirtioQCArg *arg)
//{
//	if(__builtin_expect(!!(fd==-1), 0))
//		open_device();

//	ioctl(fd, cmd, arg);
//}

////////////////////////////////////////////////////////////////////////////////
/// Module & Execution control
////////////////////////////////////////////////////////////////////////////////

void __cudaInitModule()
{

}

void** __cudaRegisterFatBinary(void *fatCubin)
{
	open_device();

	unsigned int magic;
	void **fatCubinHandle;
	time_init();
	pfunc();
	time_begin();

	magic = *(unsigned int*)fatCubin;
	fatCubinHandle = malloc(sizeof(void*)); //original
//	fatCubinHandle = myfat; //cocotion

	if( magic == FATBINC_MAGIC)
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

	send_cmd_to_device( VIRTQC_cudaRegisterFatBinary, NULL);

	// the pointer value is cubin ELF entry point
	time_end(t_RegFatbin);

	return fatCubinHandle;
}


void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	pfunc();
	time_begin();

	ptrace("fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	send_cmd_to_device( VIRTQC_cudaUnregisterFatBinary, NULL);

	free(fatCubinHandle);
	time_end(t_UnregFatbin);
	close_device();
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
	VirtioQCArg arg;
	computeFatBinaryFormat_t fatBinHeader;
	pfunc();
	time_begin();

	ptrace("fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	ptrace("hostFun= %s (%p)\n", hostFun, hostFun);
	ptrace("deviceFun= %s (%p)\n", deviceFun, deviceFun);
	ptrace("deviceName= %s\n", deviceName);
	ptrace("thread_limit= %d\n", thread_limit);

	if(tid) ptrace("tid= %u %u %u\n", tid->x, tid->y, tid->z);
	else	ptrace("tid is NULL\n");

	if(bid)	ptrace("bid= %u %u %u\n", bid->x, bid->y, bid->z);
	else	ptrace("bid is NULL\n");

	if(bDim)ptrace("bDim= %u %u %u\n", bDim->x, bDim->y, bDim->z);
	else	ptrace("bDim is NULL\n");

	if(gDim)ptrace("gDim= %u %u %u\n", gDim->x, gDim->y, gDim->z);
	else	ptrace("gDim is NULL\n");

	if(wSize)ptrace("wSize= %d\n", *wSize);
	else	 ptrace("wSize is NULL\n");

	memset(&arg, 0, sizeof(VirtioQCArg));
	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);

	ptr( arg.pA , fatBinHeader, fatBinHeader->fatSize);
	ptr( arg.pB , deviceName  , strlen(deviceName)+1 );
	arg.flag = (uint32_t)(uint64_t)hostFun;

		ptrace("pA= %p, pASize= %u, pB= %p, pBSize= %u\n", 
				(void*)arg.pA, arg.pASize, (void*)arg.pB, arg.pBSize);

	send_cmd_to_device( VIRTQC_cudaRegisterFunction, &arg);
	time_end(t_RegFunc);
}

cudaError_t cudaConfigureCall(
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
	
	cudaKernelConf[7] = (stream==NULL)?(uint64_t)-1:(uint64_t)stream;


	memset(cudaKernelPara, 0, cudaKernelParaMaxSize);
	cudaParaSize = sizeof(uint32_t);

	time_end(t_ConfigCall);
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(
		const void *arg, 
		size_t size, 
		size_t offset)
{
	pfunc();
/*
	cudaKernelPara:
	     uint32_t      uint32_t                   uint32_t
	=============================================================================
	| number of arg | arg1 size |  arg1 data  |  arg2 size  |  arg2 data  | .....
	=============================================================================
*/
	// set data size
	memcpy(&cudaKernelPara[cudaParaSize], &size, sizeof(uint32_t));
	ptrace("size= %u\n", *(uint32_t*)&cudaKernelPara[cudaParaSize]);
	cudaParaSize += sizeof(uint32_t);

	// set data 
	memcpy(&cudaKernelPara[cudaParaSize], arg, size);
	ptrace("value= %llx\n", *(unsigned long long*)&cudaKernelPara[cudaParaSize]);
	cudaParaSize += size;

	(*((uint32_t*)cudaKernelPara))++;

	time_end(t_SetArg);
	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
//	ptr( arg.pA, cudaKernelConf, 7*sizeof(uint32_t));
	ptr( arg.pA, cudaKernelConf, 8*sizeof(uint64_t));
	ptr( arg.pB, cudaKernelPara, cudaParaSize);
	arg.flag = (uint32_t)(uint64_t)func;

	send_cmd_to_device( VIRTQC_cudaLaunch, &arg);

	time_end(t_Launch);
	return cudaSuccess;
}

////////////////////////////////////////////////////////////////////////////////
/// Memory Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr( arg.pA, 0,  0);
	arg.flag = size;

	send_cmd_to_device( VIRTQC_cudaMalloc, &arg);
	*devPtr = (void*)arg.pA;
	ptrace("devPtr= %p\n", (void*)arg.pA);

	time_end(t_Malloc);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)	
{
	VirtioQCArg arg;

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr( arg.pA, devPtr, count);
	arg.para = value;

	send_cmd_to_device( VIRTQC_cudaMemset, &arg);

	return (cudaError_t)arg.cmd;
}

cudaError_t cudaFree(void* devPtr)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr( arg.pA, devPtr, 0);

	send_cmd_to_device( VIRTQC_cudaFree, &arg);
	ptrace("devPtr= %p\n", (void*)arg.pA);

	time_end(t_Free);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaMemcpy(
		void* dst, 
		const void* src, 
		size_t count,  
		enum cudaMemcpyKind kind)
{
	VirtioQCArg arg;
	
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));
	ptrace("dst= %p , src= %p ,size= %lu\n", (void*)dst, (void*)src, count);

	if( kind == cudaMemcpyHostToDevice)
	{
		ptr( arg.pA, dst, 0);
		ptr( arg.pB, src, count);
		arg.flag   = 1;
	}
	else if( kind == cudaMemcpyDeviceToHost )
	{
		ptr( arg.pA, dst, count);
		ptr( arg.pB, src, 0);
		arg.flag   = 2;
	}
	else if( kind == cudaMemcpyDeviceToDevice )
	{
		ptr( arg.pA, dst, 0);
		ptr( arg.pB, src, count);
		arg.flag   = 3;
	}
	else
	{
		error("Not impletment cudaMemcpyKind %d\n", kind);
		return cudaErrorInvalidValue;
	}

	send_cmd_to_device( VIRTQC_cudaMemcpy, &arg);

	if(kind==1){
		time_end(t_MemcpyH2D);
	}else if(kind==2){
		time_end(t_MemcpyD2H);
	}
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaMemcpyAsync(
		void * 	dst,
		const void * 	src,
		size_t 	count,
		enum cudaMemcpyKind 	kind,
		cudaStream_t 	stream)		
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));

	uint64_t mystream = (stream==NULL)?(uint64_t)-1:(uint64_t)stream;
	
	if( kind == cudaMemcpyHostToDevice)
	{
		ptr( arg.pA, dst, 0);
		ptr( arg.pB, src, count);
		//arg.rnd = (uint64_t)stream;
		arg.rnd = mystream;
		arg.flag   = 1;
	}
	else if( kind == cudaMemcpyDeviceToHost )
	{
		ptr( arg.pA, dst, count);
		ptr( arg.pB, src, 0);
		//arg.rnd = (uint64_t)stream;
		arg.rnd = mystream;
		arg.flag   = 2;
	}
	else if( kind == cudaMemcpyDeviceToDevice )
	{
		ptr( arg.pA, dst, 0);
		ptr( arg.pB, src, count);
		//arg.rnd = (uint64_t)stream;
		arg.rnd = mystream;
		arg.flag   = 3;
	}
	else
	{
		error("Not impletment cudaMemcpyKind %d\n", kind);
		return cudaErrorInvalidValue;
	}
	
	send_cmd_to_device( VIRTQC_cudaMemcpyAsync, &arg);

	return (cudaError_t)arg.cmd;
}


////////////////////////////////////////////////////////////////////////////////
/// Device Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaGetDevice(int *device)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaGetDevice, &arg);
	*device = (int)arg.pA;

	time_end(t_GetDev);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetDeviceCount(int *count)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaGetDeviceCount, &arg);
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
cudaError_t cudaSetDevice(int device)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, device, 0);
	send_cmd_to_device( VIRTQC_cudaSetDevice, &arg);

	time_end(t_SetDev);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, prop, sizeof(struct cudaDeviceProp));
	ptr( arg.pB, device, 0);
	send_cmd_to_device( VIRTQC_cudaGetDeviceProperties, &arg);

	time_end(t_GetDevProp);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaDeviceSynchronize(void)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaDeviceSynchronize, &arg);

	time_end(t_DevSync);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaDeviceReset(void)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaDeviceReset, &arg);

	time_end(t_DevReset);
	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Version Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaDriverGetVersion(int *driverVersion)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaDriverGetVersion, &arg);
	*driverVersion = (int)arg.pA;

	time_end(t_DriverGetVersion);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaRuntimeGetVersion, &arg);
	*runtimeVersion = (uint64_t)arg.pA;

	time_end(t_RuntimeGetVersion);
	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Event Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaEventCreate, &arg);
	
	*event = (void*)arg.pA;

	time_end(t_EventCreate);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)	
{
	VirtioQCArg arg;

	memset(&arg, 0, sizeof(VirtioQCArg));

	arg.flag = flags;
	send_cmd_to_device( VIRTQC_cudaEventCreateWithFlags, &arg);

	*event = (void*)arg.pA; 

	return (cudaError_t)arg.cmd;
}


cudaError_t cudaEventRecord	(cudaEvent_t event,	cudaStream_t stream)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	uint64_t mystream = (stream==NULL)?(uint64_t)-1:(uint64_t)stream;

	ptr( arg.pA, event, 0);
	//ptr( arg.pB, stream, 0);
	ptr( arg.pB, mystream, 0);
	send_cmd_to_device( VIRTQC_cudaEventRecord, &arg);

	time_end(t_EventRecord);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, event, 0);
	send_cmd_to_device( VIRTQC_cudaEventSynchronize, &arg);

	time_end(t_EventSync);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventElapsedTime(float *ms,	cudaEvent_t start, cudaEvent_t end)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, start, 0);
	ptr( arg.pB, end, 0);
	send_cmd_to_device( VIRTQC_cudaEventElapsedTime, &arg);

	memcpy(ms, &arg.flag, sizeof(float));

	time_end(t_EventElapsedTime);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, event, 0);
	send_cmd_to_device( VIRTQC_cudaEventDestroy, &arg);

	time_end(t_EventDestroy);
	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Error Handling
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaGetLastError(void)
{
	VirtioQCArg arg;
	pfunc();
	time_begin();

	memset(&arg, 0, sizeof(VirtioQCArg));

	send_cmd_to_device( VIRTQC_cudaGetLastError, &arg);

	//TODO:
	//fix register mem
    if(arg.cmd == 62) 
    {   
    	printf("***qCUDA warning***\n");
        printf("\n");
        printf("cudaErrorHostMemoryNotRegistered = 62\n");
        printf("qCUDA let it pass due to the registered memory.");
        arg.cmd = 0;
    }   

	time_end(t_GetLastError);
	return (cudaError_t)arg.cmd;
}

const char* cudaGetErrorString(cudaError_t 	error)
{
	return "Not yet implement";
}

////////about zero-copy

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));
	ptr( arg.pA, ptr, size);
	arg.flag = flags;
	
	send_cmd_to_device( VIRTQC_cudaHostRegister, &arg);
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaHostGetDevicePointer(void ** pDevice, void *pHost, unsigned int flags)	
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, 0,  0);
	ptr( arg.pB, pHost,  0);
	arg.flag = flags;

	send_cmd_to_device( VIRTQC_cudaHostGetDevicePointer, &arg);
	*pDevice = (void*)arg.pA;

	return (cudaError_t)arg.cmd;
}

cudaError_t cudaHostUnregister(void *ptr) 	
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, ptr,  0);

	send_cmd_to_device( VIRTQC_cudaHostUnregister, &arg);

	return (cudaError_t)arg.cmd;
}

/////////////////////////////////////
//about stream
/////////////////////////////////////
cudaError_t cudaStreamCreate(cudaStream_t *pStream) 
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));
	//ptr( arg.pA, pStream,  0);
	send_cmd_to_device( VIRTQC_cudaStreamCreate, &arg);

	*pStream = (cudaStream_t)arg.pA;
	return (cudaError_t)arg.cmd;

}

cudaError_t cudaStreamDestroy(cudaStream_t stream) 
{
	VirtioQCArg arg;

	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, stream, 0);
	send_cmd_to_device( VIRTQC_cudaStreamDestroy, &arg);

	return (cudaError_t)arg.cmd;

}	

// Macro to aligned up to the memory size in question                                                         
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	void *a_UA = malloc(size + MEMORY_ALIGNMENT);
	*pHost = (void *) ALIGN_UP(a_UA, MEMORY_ALIGNMENT);

	return cudaHostRegister(*pHost, size, flags);
}

cudaError_t cudaMallocHost(void **pHost, size_t size)
{
	return cudaHostAlloc(pHost, size, cudaHostAllocDefault);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) 	
{
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));
	arg.flag = flags;
	send_cmd_to_device( VIRTQC_cudaSetDeviceFlags, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaFreeHost(void *ptr) 	
{
	cudaHostUnregister(ptr);
	VirtioQCArg arg;
	memset(&arg, 0, sizeof(VirtioQCArg));
	
	ptr(arg.pA, ptr, 0);
	send_cmd_to_device( VIRTQC_cudaFreeHost, &arg);
	return (cudaError_t)arg.cmd;
}


