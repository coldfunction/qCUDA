#include "qemu-common.h"
#include "qemu/iov.h"
#include "qemu/error-report.h"
#include "hw/virtio/virtio.h"
#include "hw/virtio/virtio-bus.h"
#include "hw/virtio/virtio-qcuda.h"
#include <sys/mman.h>

#ifdef CONFIG_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#endif

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


#include "../../../qcu-driver/qcuda_common.h"

#define error(fmt, arg...) \
	error_report("file %s ,line %d ,ERROR: "fmt, __FILE__, __LINE__, ##arg)

#ifndef MIN
#define MIN(a,b) ({ ((a)<(b))? (a):(b) })
#endif

#define VIRTHM_DEV_PATH "/dev/vf0"

uint32_t BLOCK_SIZE;

char *deviceSpace = NULL;
uint32_t deviceSpaceSize = 0;


static void* gpa_to_hva(uint64_t pa) 
{
	MemoryRegionSection section;

	section = memory_region_find(get_system_memory(), (ram_addr_t)pa, 1);
	if ( !int128_nz(section.size) || !memory_region_is_ram(section.mr)){
		error("addr %p in rom\n", (void*)pa); 
		return 0;
	}

	return (memory_region_get_ram_ptr(section.mr) + section.offset_within_region);
}

#ifdef CONFIG_CUDA
CUdevice cudaDeviceCurrent;
CUcontext cudaContext;

int cudaContext_count;

#define cudaFunctionMaxNum 8
uint32_t cudaFunctionNum;

#define cudaEventMaxNum 16
cudaEvent_t cudaEvent[cudaEventMaxNum];
uint32_t cudaEventNum;

#define cudaStreamMaxNum 32
cudaStream_t cudaStream[cudaStreamMaxNum];
uint32_t cudaStreamNum;

typedef struct kernelInfo
{
	void *fatBin;
	char functionName[50];
	uint32_t funcId;

}kernelInfo;

typedef struct cudaDev
{
	CUdevice device;
	CUcontext context;
	uint32_t cudaFunctionId[cudaFunctionMaxNum];
	CUfunction cudaFunction[cudaFunctionMaxNum];
	CUmodule module;
	int kernelsLoaded;	
}cudaDev;

int totalDevices;
cudaDev *cudaDevices;
cudaDev zeroedDevice;
kernelInfo devicesKernels[cudaFunctionMaxNum];



#define cudaError(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(cudaError_t err, const int line)
{
	char *str;
	if ( err != cudaSuccess )
	{
		str = (char*)cudaGetErrorString(err);
		error_report("CUDA Runtime API error = %04d \"%s\" line %d\n", err, str, line);
	}
}


#define cuError(err)  __cuErrorCheck(err, __LINE__) 
static inline void __cuErrorCheck(CUresult err, const int line)
{
	char *str;
	if ( err != CUDA_SUCCESS )
	{   
		cuGetErrorName(err, (const char**)&str);
		error_report("CUDA Runtime API error = %04d \"%s\" line %d\n", err, str, line);
	}   
}


static void loadModuleKernels(int devId, void *fBin, char *fName,  uint32_t fId, uint32_t fNum)
{
	pfunc();
	ptrace("loading module.... fatBin= %16p ,name= '%s', fId = '%d'\n", fBin, fName, fId);
	cuError( cuModuleLoadData( &cudaDevices[devId].module, fBin ));
	cuError( cuModuleGetFunction(&cudaDevices[devId].cudaFunction[fNum],
									cudaDevices[devId].module, fName) );
 	cudaDevices[devId].cudaFunctionId[fNum] = fId;
	cudaDevices[devId].kernelsLoaded = 1;
}

static void reloadAllKernels(void)
{
	pfunc();
	void *fb;
	char *fn;
	uint32_t i = 0, fId;

	for(i = 0; i < cudaFunctionNum; i++)
	{
		fb = devicesKernels[i].fatBin;
		fn = devicesKernels[i].functionName;
		fId = devicesKernels[i].funcId;

 		loadModuleKernels( cudaDeviceCurrent, fb, fn, fId, i );
	}	
}

static cudaError_t initializeDevice(void)
{
	int device = cudaDeviceCurrent;
	if( device >= totalDevices )
	{
		ptrace("error setting device= %d\n", device);
		return cudaErrorInvalidDevice;
	}
	else
	{
		// device was reset therefore no context
		if( !memcmp( &zeroedDevice, &cudaDevices[device], sizeof(cudaDev) ) ) 
		{
			cuError( cuDeviceGet(&cudaDevices[device].device, device) );
			cuError( cuCtxCreate(&cudaDevices[device].context, 0, cudaDevices[device].device) );
		}
		else
		{
			cuError( cuCtxSetCurrent(cudaDevices[device].context) );
			ptrace("cuda device %d\n", cudaDevices[device].device);
		}
		if( cudaDevices[device].kernelsLoaded == 0 )
			reloadAllKernels();
		return 0;
	}

}



////////////////////////////////////////////////////////////////////////////////
///	Module & Execution control (driver API)
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaRegisterFatBinary(VirtioQCArg *arg)
{
	uint32_t i;
	pfunc();

	for(i=0; i<cudaEventMaxNum; i++)
		memset(&cudaEvent[i], 0, sizeof(cudaEvent_t));

	for(i=0; i<cudaStreamMaxNum; i++)
		memset(&cudaStream[i], 0, sizeof(cudaStream_t));

	cuError( cuInit(0) );

	cuError( cuDeviceGetCount(&totalDevices) );
	cudaDevices = (cudaDev *) malloc(totalDevices * sizeof(cudaDev));
	memset(&zeroedDevice, 0, sizeof(cudaDev));

	i = totalDevices;
	// the last created context is the one used & associated with the device
	// so do this in reverse order
	while(i-- != 0)
	{
		ptrace("creating context for device %d\n", i);
		memset(&cudaDevices[i], 0, sizeof(cudaDev));
		cuError( cuDeviceGet(&cudaDevices[i].device, i) );
		cuError( cuCtxCreate(&cudaDevices[i].context, 0, cudaDevices[i].device) );
		memset(&cudaDevices[i].cudaFunction, 0, sizeof(CUfunction) * cudaFunctionMaxNum);
 		cudaDevices[i].kernelsLoaded = 0;
	}

	cudaDeviceCurrent = cudaDevices[0].device; // used when calling cudaGetDevice
	
	cudaFunctionNum = 0;
	cudaEventNum = 0;

	cudaStreamNum = 1;
	
	cudaContext_count = 1;
}

static void qcu_cudaUnregisterFatBinary(VirtioQCArg *arg)
{
	uint32_t i;
	pfunc();

	//for(i=0; i<cudaEventMaxNum; i++)
	//{
		//if( cudaEvent[i] != 0 ){
		//	cudaError( cudaEventDestroy(cudaEvent[i]));
		//}
	//}

	for(i = 0; i < totalDevices; i++)
	{
		// get rid of default context if any
		// when a device is reset there will be no context
		if( memcmp( &zeroedDevice, &cudaDevices[i], sizeof(cudaDev) ) != 0 )
		{
			cudaError( cuCtxDestroy(cudaDevices[i].context) );
		}
	}
	free(cudaDevices);
}

static void qcu_cudaRegisterFunction(VirtioQCArg *arg)
{
	void *fatBin;
	char *functionName;
	uint32_t funcId;
	pfunc();

	// assume fatbin size is less equal 4MB
	fatBin       = gpa_to_hva(arg->pA);
	functionName = gpa_to_hva(arg->pB);
	funcId		 = arg->flag;

	//initialize the kernelInfo
	devicesKernels[cudaFunctionNum].fatBin = malloc(4*1024*1024);

	memcpy(devicesKernels[cudaFunctionNum].fatBin, fatBin, arg->pASize);
	memcpy(devicesKernels[cudaFunctionNum].functionName, functionName, arg->pBSize);
	devicesKernels[cudaFunctionNum].funcId = funcId;

	ptrace("fatBin= %16p ,name= '%s'\n", fatBin, functionName);

	loadModuleKernels( cudaDeviceCurrent, fatBin, functionName, funcId, cudaFunctionNum );

	cudaFunctionNum++;
}

static void qcu_cudaLaunch(VirtioQCArg *arg)
{
	uint64_t *conf;
	uint8_t *para;
	uint32_t funcId, paraNum, paraIdx, funcIdx;
	void **paraBuf;
	int i;
	pfunc();

	conf = gpa_to_hva(arg->pA);
	para = gpa_to_hva(arg->pB);
	paraNum = *((uint32_t*)para);
	funcId = arg->flag;
	
	ptrace("paraNum= %u\n", paraNum);

	paraBuf = malloc(paraNum*sizeof(void*));
	paraIdx = sizeof(uint32_t);

	for(i=0; i<paraNum; i++)
	{
		paraBuf[i] = &para[paraIdx+sizeof(uint32_t)];
		ptrace("arg %d = 0x%llx size= %u byte\n", i, 
			*(unsigned long long*)paraBuf[i], *(unsigned int*)&para[paraIdx]);

		paraIdx += *((uint32_t*)&para[paraIdx]) + sizeof(uint32_t);
	}

	for(funcIdx=0; funcIdx<cudaFunctionNum; funcIdx++)
	{
		if( cudaDevices[cudaDeviceCurrent].cudaFunctionId[funcIdx] == funcId )
			break;
	}

	ptrace("grid (%u %u %u) block(%u %u %u) sharedMem(%u)\n", 
			conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6]);

	cuError( cuLaunchKernel(cudaDevices[cudaDeviceCurrent].cudaFunction[funcIdx],
				conf[0], conf[1], conf[2],
				conf[3], conf[4], conf[5], 
				conf[6], (conf[7]==(uint64_t)-1)?NULL:cudaStream[conf[7]], paraBuf, NULL)); 
	
	free(paraBuf);
}

////////////////////////////////////////////////////////////////////////////////
/// Memory Management (runtime API)
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaMalloc(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t count;
	void* devPtr;
	pfunc();

	// in case cudaReset was the previous call
	initializeDevice(); 


	count = arg->flag;
	cudaError((err = cudaMalloc( &devPtr, count )));
	arg->cmd = err;
	arg->pA = (uint64_t)devPtr;

	ptrace("ptr= %p ,count= %u\n", (void*)arg->pA, count);
}

static void qcu_cudaMemset(VirtioQCArg *arg)
{
	cudaError_t err;
	void* dst;

	// in case cudaReset was the previous call
	initializeDevice(); 

	dst = (void*)arg->pA;
	cudaError((err = cudaMemset(dst, arg->para, arg->pASize)));
	arg->cmd = err;
}

static void qcu_cudaMemcpy(VirtioQCArg *arg)
{
	
	cudaError_t err = 0;
	uint32_t size, len, i;

	void *dst, *src;
	uint64_t *gpa_array;

	pfunc();

	initializeDevice();

	if( arg->flag == cudaMemcpyHostToDevice )
	{
		dst = (void*)arg->pA;
		size = arg->pBSize;
#ifdef USER_KERNEL_COPY		
		if( size > QCU_KMALLOC_MAX_SIZE)
		{
#endif
   			if(arg->para)	     	
			{
				src = gpa_to_hva(arg->pB);
				err = cuMemcpyHtoD((CUdeviceptr)dst, src, size);
			}
			else
			{
				gpa_array = gpa_to_hva(arg->pB);
		
				uint32_t offset   	 = arg->pASize;
				uint32_t start_offset = offset%BLOCK_SIZE;
				uint32_t rsize = BLOCK_SIZE - start_offset;
            
				src = gpa_to_hva(gpa_array[0]);
            	len = MIN(size, rsize);
				err = cuMemcpyHtoD((CUdeviceptr)dst, src, len);
	
				size -= len;
				dst += len;

            	for(i=0; size>0; i++)
            	{   
            		src = gpa_to_hva(gpa_array[i+1]);
                	len = MIN(size, BLOCK_SIZE);
					err = cuMemcpyHtoD((CUdeviceptr)dst, src, len);
                
					size -= len;
                	dst  += len;
            	}               

			}
        
#ifdef USER_KERNEL_COPY		
		}
		else
		{
			src = gpa_to_hva(arg->pB);
			err = cuMemcpyHtoD((CUdeviceptr)dst, src, size);
		}
#endif

	}
	else if(arg->flag == cudaMemcpyDeviceToHost )
	{
		src = (void*)arg->pB;
		size = arg->pASize;

#ifdef USER_KERNEL_COPY		
		if( size > QCU_KMALLOC_MAX_SIZE)
		{
#endif
			/*		
			fd 	   = ldl_p(&arg->pBSize);
			offset = arg->pA;	

    		dst = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
			err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
			*/

			if(arg->para)
			{
				dst = gpa_to_hva(arg->pA);
				err = cuMemcpyDtoH(dst, (CUdeviceptr)src, size);
			}
			else
			{
        		gpa_array = gpa_to_hva(arg->pA);
		
				uint32_t offset   	 = arg->pBSize;
				uint32_t start_offset = offset%BLOCK_SIZE;
				uint32_t rsize = BLOCK_SIZE - start_offset;

				dst = gpa_to_hva(gpa_array[0]);
            	len = MIN(size, rsize);
				err = cuMemcpyDtoH(dst, (CUdeviceptr)src, len);
			
				size -= len;
				src+=len;	
            
				for(i=0; size>0; i++)
  	            {
     		       	dst = gpa_to_hva(gpa_array[i+1]);
                	len = MIN(size, BLOCK_SIZE);
					err = cuMemcpyDtoH(dst, (CUdeviceptr)src, len);
                
					size -= len;
              		src  += len;
            	}   
			}                    
#ifdef USER_KERNEL_COPY		
		}
		else
		{
			dst = gpa_to_hva(arg->pA);
			err = cuMemcpyDtoH(dst, (CUdeviceptr)src, size);
		}
#endif
	}
	else if( arg->flag == cudaMemcpyDeviceToDevice )
	{
		dst = (void*)arg->pA;
		src = (void*)arg->pB;
		size = arg->pBSize;
		cudaError(( err = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, size)));
	}
	
	arg->cmd = err;
	ptrace("size= %u\n", size);
}

static void qcu_cudaMemcpyAsync(VirtioQCArg *arg)
{

	uint32_t size, len, i;
	cudaError_t err = 0;
	void *dst, *src;
	uint64_t *gpa_array;
	
	uint64_t streamIdx = arg->rnd;
	cudaStream_t stream = (streamIdx==(uint64_t)-1)?NULL:cudaStream[streamIdx];
	
	if( arg->flag == cudaMemcpyHostToDevice )
	{
		dst = (void*)arg->pA;
		size = arg->pBSize;
		gpa_array = gpa_to_hva(arg->pB);
		
		uint32_t offset   	 = arg->pASize;
		uint32_t start_offset = offset%BLOCK_SIZE;
		uint32_t rsize = BLOCK_SIZE - start_offset;
            
		src = gpa_to_hva(gpa_array[0]);
        len = MIN(size, rsize);
		err = cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, stream);
	
		size -= len;
		dst += len;

        for(i=0; size>0; i++)
        {   
        	src = gpa_to_hva(gpa_array[i+1]);
            len = MIN(size, BLOCK_SIZE);
			err = cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, stream);
			size -= len;
            dst  += len;
        }
	}
	else if(arg->flag == cudaMemcpyDeviceToHost )
	{
		src = (void*)arg->pB;
		size = arg->pASize;
        
		gpa_array = gpa_to_hva(arg->pA);
		
		uint32_t offset   	 = arg->pBSize;
		uint32_t start_offset = offset%BLOCK_SIZE;
		uint32_t rsize = BLOCK_SIZE - start_offset;

		dst = gpa_to_hva(gpa_array[0]);
        len = MIN(size, rsize);
		err = cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, stream);
			
		size -= len;
		src+=len;	
            
		for(i=0; size>0; i++)
  	    {
     		dst = gpa_to_hva(gpa_array[i+1]);
            len = MIN(size, BLOCK_SIZE);
			err = cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, stream);
                
			size -= len;
            src  += len;
        }   
	}
	else if( arg->flag == cudaMemcpyDeviceToDevice )
	{
		src = (void*)arg->pA;
		dst = (void*)arg->pB;
		size = arg->pBSize;
		cudaError(( err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream)));
	}
	
	arg->cmd = err;
	
}

static void qcu_cudaFree(VirtioQCArg *arg)
{
	cudaError_t err;
	void* dst;
	pfunc();

	initializeDevice();

	dst = (void*)arg->pA;
	cudaError((err = cudaFree(dst)));
	arg->cmd = err;

	ptrace("ptr= %16p\n", dst);
}

////////////////////////////////////////////////////////////////////////////////
///	Device Management
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaGetDevice(VirtioQCArg *arg)
{
	cudaError_t err;
	pfunc();

	initializeDevice();

	err = 0;
	arg->cmd = err;
	arg->pA = (uint64_t)cudaDevices[cudaDeviceCurrent].device;
}

static void qcu_cudaGetDeviceCount(VirtioQCArg *arg)
{
	cudaError_t err;
	int device;
	pfunc();

	initializeDevice();

	cudaError((err = cudaGetDeviceCount( &device )));
	arg->cmd = err;
	arg->pA = (uint64_t)device;

	ptrace("device count=%d\n", device);
}

static void qcu_cudaSetDevice(VirtioQCArg *arg)
{
	cudaError_t err;
	int device;
	pfunc();

	device = (int)arg->pA;

	cudaDeviceCurrent = device;
	cudaError((err = initializeDevice()));

	arg->cmd = err;

	ptrace("set device= %d\n", device);
}

static void qcu_cudaGetDeviceProperties(VirtioQCArg *arg)
{
	cudaError_t err;
	struct cudaDeviceProp *prop;
	int device;
	pfunc();

	prop = gpa_to_hva(arg->pA);
	device = (int)arg->pB;

	cudaError((err = cudaGetDeviceProperties( prop, device )));
	arg->cmd = err;

	ptrace("get prop for device %d\n", device);
}

static void qcu_cudaDeviceSynchronize(VirtioQCArg *arg)
{
	cudaError_t err;
	pfunc();
	cudaError((err = cudaDeviceSynchronize()));
	arg->cmd = err;
}

static void qcu_cudaDeviceReset(VirtioQCArg *arg)
{
	cudaError_t err;
	pfunc();

	// TODO: 
	// should get rid of events for current device
	cuCtxDestroy(cudaDevices[cudaDeviceCurrent].context);

	cudaError((err = cudaDeviceReset()));

	memset( &cudaDevices[cudaDeviceCurrent], 0, sizeof(cudaDev) );
	arg->cmd = err;
}

////////////////////////////////////////////////////////////////////////////////
///	Version Management
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaDriverGetVersion(VirtioQCArg *arg)
{
	cudaError_t err;
	int version;
	pfunc();

	cudaError((err = cudaDriverGetVersion( &version )));
	arg->cmd = err;
	arg->pA = (uint64_t)version;

	ptrace("driver version= %d\n", version);
}

static void qcu_cudaRuntimeGetVersion(VirtioQCArg *arg)
{
	cudaError_t err;
	int version;
	pfunc();

	cudaError((err = cudaRuntimeGetVersion( &version )));
	arg->cmd = err;
	arg->pA = (uint64_t)version;

	ptrace("runtime driver= %d\n", version);
}

//////////////////////////////////////////////////
//static void qcu_checkCudaCapabilities(VirtioQCArg *arg)
//{
//	cudaError_t err;
//	err = checkCudaCapabilities(arg->pA, arg->pB);
//	arg->cmd = err;
//}

//////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///	Event Management
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaEventCreate(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;
	pfunc();

	idx = cudaEventNum;
	cudaError((err = cudaEventCreate(&cudaEvent[idx])));
	arg->cmd = err;
	arg->pA = (uint64_t)idx;

	cudaEventNum++;
	ptrace("create event %u\n", idx);
}

static void qcu_cudaEventCreateWithFlags(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;

	idx = cudaEventNum;
	cudaError((err = cudaEventCreateWithFlags(&cudaEvent[idx], arg->flag)));
	arg->cmd = err;
	arg->pA = (uint64_t)idx;

	cudaEventNum++;
	ptrace("create event %u\n", idx);
}

static void qcu_cudaEventRecord(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t eventIdx;
	uint64_t streamIdx;
	pfunc();

	eventIdx  = arg->pA;
	streamIdx = arg->pB;
	cudaError((err = cudaEventRecord(cudaEvent[eventIdx], (streamIdx==(uint64_t)-1)?NULL:cudaStream[streamIdx]  )));
	arg->cmd = err;

	ptrace("event record %u\n", eventIdx);
}

static void qcu_cudaEventSynchronize(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;
	pfunc();

	idx = arg->pA;
	cudaError((err = cudaEventSynchronize( cudaEvent[idx] )));
	arg->cmd = err;

	ptrace("sync event %u\n", idx);
}

static void qcu_cudaEventElapsedTime(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t startIdx;
	uint32_t endIdx;
	float ms;
	pfunc();

	startIdx = arg->pA;
	endIdx   = arg->pB;
	cudaError((err = cudaEventElapsedTime(&ms, cudaEvent[startIdx], cudaEvent[endIdx])));
	arg->cmd = err;
	memcpy(&arg->flag, &ms, sizeof(float));

	ptrace("event elapse time= %f, start= %u, end= %u\n", 
			ms, startIdx, endIdx);
}

static void qcu_cudaEventDestroy(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;
	pfunc();

	idx = arg->pA;
	cudaError((err = cudaEventDestroy(cudaEvent[idx])));
	arg->cmd = err;

	ptrace("destroy event %u\n", idx);
}

////////////////////////////////////////////////////////////////////////////////
///	Error Handling
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaGetLastError(VirtioQCArg *arg)
{
	cudaError_t err;
	pfunc();

	err =  cudaGetLastError();
	arg->cmd = err;
	ptrace("lasr cudaError %d\n", err);
}

//////////zero-copy////////

static void qcu_cudaHostRegister(VirtioQCArg *arg)
{
	int size, i;
	cudaError_t err = 0;
	void *ptr;
	uint64_t *gpa_array;
	size = arg->pASize;
	gpa_array = gpa_to_hva(arg->pA);
	
	for(i = 0; size>0; i++)
	{
    	ptr = gpa_to_hva(gpa_array[i]);
		err = cudaHostRegister(ptr, MIN(BLOCK_SIZE,size), arg->flag);
		size-=BLOCK_SIZE;	
	}
	arg->cmd = err;

	//fix for cudaHostGetDevicePointer start
	int fd = ldl_p(&arg->pBSize);
	uint64_t offset = arg->rnd;	
	size = arg->pASize;
	
	ptr = mmap(0, arg->pB, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
 	
	err = cudaHostRegister(ptr, size, arg->flag);
	arg->rnd = (uint64_t)ptr;
	arg->cmd = err;
	//fix for cudaHostGetDevicePointer end
}

static void qcu_cudaHostGetDevicePointer(VirtioQCArg *arg)
{
	cudaError_t err;
	void *ptr = (void*)	arg->pB;
	void *devPtr;
	err = cudaHostGetDevicePointer(&devPtr, ptr, arg->flag);		
	arg->pA = (uint64_t)devPtr;
	arg->cmd = err;
}

static void qcu_cudaHostUnregister(VirtioQCArg *arg)
{
		
	uint32_t size, i;
	void *ptr;
	uint64_t *gpa_array;
	cudaError_t err = 0;
	size = arg->pASize;
	
	gpa_array = gpa_to_hva(arg->pA);

	for(i = 0; size>0; i++)
	{
    	ptr = gpa_to_hva(gpa_array[i]);
		size-=BLOCK_SIZE;	
		err = cudaHostUnregister(ptr);	
	}
	arg->cmd = err; 	
}

static void qcu_cudaSetDeviceFlags(VirtioQCArg *arg)
{
	cudaError_t err;
//	err = cudaSetDeviceFlags(arg->flag);
	err = 0;
	arg->cmd = err;
}

//stream
static void qcu_cudaStreamCreate(VirtioQCArg *arg)
{
	cudaError_t err;

	err = cudaStreamCreate(&cudaStream[cudaStreamNum]);
	arg->pA = cudaStreamNum;
	cudaStreamNum++;
 	arg->cmd = err;
}

static void qcu_cudaStreamDestroy(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;

	idx = arg->pA;
	cudaError((err = cudaStreamDestroy(cudaStream[idx])));
	arg->cmd = err;
	memset(&cudaStream[idx], 0, sizeof(cudaStream_t));
}

#endif // CONFIG_CUDA

static int qcu_cmd_write(VirtioQCArg *arg)
{
	void   *src, *dst;
	uint64_t *gpa_array;
	uint32_t size, len, i;

	size = arg->pASize;

	ptrace("szie= %u\n", size);

	if(deviceSpace!=NULL)
	{
		free(deviceSpace);	
	}

	deviceSpaceSize = size;
	deviceSpace = (char*)malloc(deviceSpaceSize);

	if( size > deviceSpaceSize )
	{
		gpa_array = gpa_to_hva(arg->pA);
		dst = deviceSpace;
		for(i=0; size>0; i++)
		{
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);
			src = gpa_to_hva(gpa_array[i]);
			memcpy(dst, src, len);
			size -= len;
			dst  += len;
		}
	}
	else
	{
		src = gpa_to_hva(arg->pA);
		memcpy(deviceSpace, src, size);
	}
	// checker ------------------------------------------------------------
/*
	uint64_t err;
	if( deviceSpaceSize<32 )
	{
		for(i=0; i<deviceSpaceSize; i++)
		{
			ptrace("deviceSpace[%lu]= %d\n", i, deviceSpace[i]);
		}
	}
	else
	{	
		err = 0;
		for(i=0; i<deviceSpaceSize; i++)
		{
			if( deviceSpace[i] != (i%17)*7 ) err++;
		}
		ptrace("error= %llu\n", (unsigned long long)err);
	}
	ptrace("\n\n");
	//---------------------------------------------------------------------
*/
	return 0;
}

static int qcu_cmd_read(VirtioQCArg *arg)
{
	void   *src, *dst;
	uint64_t *gpa_array;
	uint32_t size, len, i;

	if(deviceSpace==NULL)
	{
		return -1;
	}

	size = arg->pASize;

	ptrace("szie= %u\n", size);
	
	if( size > deviceSpaceSize )
	{
		gpa_array = gpa_to_hva(arg->pA);
		src = deviceSpace;
		for(i=0; size>0; i++)
		{
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);
			dst = gpa_to_hva(gpa_array[i]);
			memcpy(dst, src, len);
			size -= len;
			src  += len;
		}
	}
	else
	{
		dst = gpa_to_hva(arg->pA);
		memcpy(dst, deviceSpace, size);
	}

	return 0;
}

static int qcu_cmd_mmapctl(VirtioQCArg *arg)
{

	int pid = getpid();	
	char vfname[100];

	sprintf(vfname, "/dev/qcuvf/vm%d_%lx", pid, arg->pB);

	int fd=open(vfname, O_CREAT|O_RDWR,0666);
    if(fd<0)
    {   
        printf("failure to open\n");
        exit(0);
    } 
	if(lseek(fd,arg->pBSize,SEEK_SET)==-1) 
    {   
        printf("Failure to lseek\n");
        exit(0);
	}   
    if(write(fd, "",1) != 1)  
    {   
        printf("Failure on write\n");
        exit(0);
    }   
	
	stl_p(&arg->pA, fd);
	
	return 0;
}

static int qcu_cmd_open(VirtioQCArg *arg)
{
	BLOCK_SIZE = arg->pASize;

	return 0;

}

static int qcu_cmd_close(VirtioQCArg *arg)
{
	return 0;
}

static int qcu_cmd_mmap(VirtioQCArg *arg){
		
	//arg->pA: file fd
	//arg->pASize: numOfblocks
	//arg->pB: gpa_array

	int32_t fd = (int32_t)ldl_p(&arg->pA);
	uint64_t *gpa_array = gpa_to_hva(arg->pB);
	void *addr;
   	
	int i;
	for(i = 0; i < arg->pASize; i++)
	{
		addr = gpa_to_hva(gpa_array[i]);
		munmap(addr, BLOCK_SIZE);
   		mmap(addr, BLOCK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, i*BLOCK_SIZE);
	}	


    return 0;
}

static int qcu_cmd_munmap(VirtioQCArg *arg){
   	// arg->pB: gpa_array
	// arg->pBSize: numOfblocks
   
	uint64_t *gpa_array = gpa_to_hva(arg->pB);
	void *addr;
	
	int i;
	for(i = 0; i < arg->pBSize; i++)
	{
		addr    = gpa_to_hva(gpa_array[i]);
		munmap(addr, BLOCK_SIZE);
		mmap(addr, BLOCK_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
   	} 
	
	return 0;

}

static int qcu_cmd_mmaprelease(VirtioQCArg *arg)
{
	close(ldl_p(&arg->pBSize));

	//TODO: safely check
	
	int pid = getpid();	
	char vfname[100];
	sprintf(vfname, "/dev/qcuvf/vm%d_%lx", pid, arg->pA);

	unlink(vfname);

	return 0;
}

static void virtio_qcuda_cmd_handle(VirtIODevice *vdev, VirtQueue *vq)
{
	VirtQueueElement elem;
	VirtioQCArg *arg;

	arg = malloc( sizeof(VirtioQCArg));
	while( virtqueue_pop(vq, &elem) )
	{
		iov_to_buf(elem.out_sg, elem.out_num, 0, arg, sizeof(VirtioQCArg));

		switch( arg->cmd )
		{
			case VIRTQC_CMD_WRITE:
				qcu_cmd_write(arg);
				break;

			case VIRTQC_CMD_READ:
				qcu_cmd_read(arg); 
				break;

			case VIRTQC_CMD_OPEN:
				qcu_cmd_open(arg);	
				break;

			case VIRTQC_CMD_CLOSE:
				qcu_cmd_close(arg);	
				break;

			case VIRTQC_CMD_MMAP:
				qcu_cmd_mmap(arg);
				break;

			case VIRTQC_CMD_MUNMAP:
				qcu_cmd_munmap(arg);
				break;

			case VIRTQC_CMD_MMAPCTL:
				qcu_cmd_mmapctl(arg);
				break;
			
			case VIRTQC_CMD_MMAPRELEASE:	
				qcu_cmd_mmaprelease(arg);
				break;
	
#ifdef CONFIG_CUDA
			// Module & Execution control (driver API)
			case VIRTQC_cudaRegisterFatBinary:
				qcu_cudaRegisterFatBinary(arg);
				break;

			case VIRTQC_cudaUnregisterFatBinary:
				qcu_cudaUnregisterFatBinary(arg); 
				break;

			case VIRTQC_cudaRegisterFunction:
				qcu_cudaRegisterFunction(arg);
				break;

			case VIRTQC_cudaLaunch:
				qcu_cudaLaunch(arg);
				break;

			// Memory Management (runtime API)
			case VIRTQC_cudaMalloc:
				qcu_cudaMalloc(arg);
				break;
			
			case VIRTQC_cudaMemset:
				qcu_cudaMemset(arg);
				break;

			case VIRTQC_cudaMemcpy:
				qcu_cudaMemcpy(arg);
				break;
			
			case VIRTQC_cudaMemcpyAsync:
				qcu_cudaMemcpyAsync(arg);
				break;

			case VIRTQC_cudaFree:
				qcu_cudaFree(arg);
				break;

			// Device Management (runtime API)
			case VIRTQC_cudaGetDevice:
				qcu_cudaGetDevice(arg);
				break;

			case VIRTQC_cudaGetDeviceCount:
				qcu_cudaGetDeviceCount(arg);
				break;

			case VIRTQC_cudaSetDevice:
				qcu_cudaSetDevice(arg);
				break;

			case VIRTQC_cudaGetDeviceProperties:
				qcu_cudaGetDeviceProperties(arg);
				break;

			case VIRTQC_cudaDeviceSynchronize:
				qcu_cudaDeviceSynchronize(arg);
				break;

			case VIRTQC_cudaDeviceReset:
				qcu_cudaDeviceReset(arg);
				break;

			// Version Management (runtime API)
			case VIRTQC_cudaDriverGetVersion:
				qcu_cudaDriverGetVersion(arg);
				break;

			case VIRTQC_cudaRuntimeGetVersion:
				qcu_cudaRuntimeGetVersion(arg);
				break;
///////////////////////////////////////////////
		//	case VIRTQC_checkCudaCapabilities:
		//		qcu_checkCudaCapabilities(arg);
///////////////////////////////////////////////

			//stream
			case VIRTQC_cudaStreamCreate:
				qcu_cudaStreamCreate(arg);	
				break;
			
			case VIRTQC_cudaStreamDestroy:
				qcu_cudaStreamDestroy(arg);	
				break;

			// Event Management (runtime API)
			case VIRTQC_cudaEventCreate:
				qcu_cudaEventCreate(arg);
				break;
			
			case VIRTQC_cudaEventCreateWithFlags:
				qcu_cudaEventCreateWithFlags(arg);
				break;

			case VIRTQC_cudaEventRecord:
				qcu_cudaEventRecord(arg);
				break;

			case VIRTQC_cudaEventSynchronize:
				qcu_cudaEventSynchronize(arg);
				break;

			case VIRTQC_cudaEventElapsedTime:
				qcu_cudaEventElapsedTime(arg);
				break;

			case VIRTQC_cudaEventDestroy:
				qcu_cudaEventDestroy(arg);
				break;

			// Error Handling (runtime API)
			case VIRTQC_cudaGetLastError:
				qcu_cudaGetLastError(arg);
				break;

			//zero-copy
			case VIRTQC_cudaHostRegister:
				qcu_cudaHostRegister(arg);
				break;
		
			case VIRTQC_cudaHostGetDevicePointer:
				qcu_cudaHostGetDevicePointer(arg);
				break;

			case VIRTQC_cudaHostUnregister:
				qcu_cudaHostUnregister(arg);
				break;
			
			case VIRTQC_cudaSetDeviceFlags:
				qcu_cudaSetDeviceFlags(arg);
				break;
	
			//case VIRTQC_cudaFreeHost:	
			//	qcu_cudaFreeHost(arg);
			//	break;
	
#endif
			default:
				error("unknow cmd= %d\n", arg->cmd);
		}

		iov_from_buf(elem.in_sg, elem.in_num, 0, arg, sizeof(VirtioQCArg));
		virtqueue_push(vq, &elem, sizeof(VirtioQCArg));
		virtio_notify(vdev, vq);
	}
		free(arg);
}

//####################################################################
//   class basic callback functions
//####################################################################

static void virtio_qcuda_device_realize(DeviceState *dev, Error **errp)
{
	VirtIODevice *vdev = VIRTIO_DEVICE(dev);
	VirtIOQC *qcu = VIRTIO_QC(dev);
	//Error *err = NULL;

	//ptrace("GPU mem size=%"PRIu64"\n", qcu->conf.mem_size);

	virtio_init(vdev, "virtio-qcuda", VIRTIO_ID_QC, sizeof(VirtIOQCConf));

	qcu->vq  = virtio_add_queue(vdev, 1024, virtio_qcuda_cmd_handle);
}

static uint64_t virtio_qcuda_get_features(VirtIODevice *vdev, uint64_t features, Error **errp)
{
	//ptrace("feature=%"PRIu64"\n", features);
	return features;
}

/*
   static void virtio_qcuda_device_unrealize(DeviceState *dev, Error **errp)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_get_config(VirtIODevice *vdev, uint8_t *config)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_set_config(VirtIODevice *vdev, const uint8_t *config)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_reset(VirtIODevice *vdev)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_save_device(VirtIODevice *vdev, QEMUFile *f)
   {
   ptrace("\n");
   }

   static int virtio_qcuda_load_device(VirtIODevice *vdev, QEMUFile *f, int version_id)
   {
   ptrace("\n");
   return 0;
   }

   static void virtio_qcuda_set_status(VirtIODevice *vdev, uint8_t status)
   {
   ptrace("\n");
   }
 */

/*
   get the configure
ex: -device virtio-qcuda,size=2G,.....
DEFINE_PROP_SIZE(config name, device struce, element, default value)
 */
static Property virtio_qcuda_properties[] = 
{
	DEFINE_PROP_SIZE("size", VirtIOQC, conf.mem_size, 0),
	DEFINE_PROP_END_OF_LIST(),
};

static void virtio_qcuda_class_init(ObjectClass *klass, void *data)
{
	DeviceClass *dc = DEVICE_CLASS(klass);
	VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

	dc->props = virtio_qcuda_properties;

	set_bit(DEVICE_CATEGORY_MISC, dc->categories);

	vdc->get_features = virtio_qcuda_get_features;

	vdc->realize = virtio_qcuda_device_realize;
	/*	
		vdc->unrealize = virtio_qcuda_device_unrealize;

		vdc->get_config = virtio_qcuda_get_config;
		vdc->set_config = virtio_qcuda_set_config;

		vdc->save = virtio_qcuda_save_device;
		vdc->load = virtio_qcuda_load_device;

		vdc->set_status = virtio_qcuda_set_status;
		vdc->reset = virtio_qcuda_reset;
	 */	
}

static void virtio_qcuda_instance_init(Object *obj)
{
}

static const TypeInfo virtio_qcuda_device_info = {
	.name = TYPE_VIRTIO_QC,
	.parent = TYPE_VIRTIO_DEVICE,
	.instance_size = sizeof(VirtIOQC),
	.instance_init = virtio_qcuda_instance_init,
	.class_init = virtio_qcuda_class_init,
};

static void virtio_qcuda_register_types(void)
{
	type_register_static(&virtio_qcuda_device_info);
}

type_init(virtio_qcuda_register_types)
