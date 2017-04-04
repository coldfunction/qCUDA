#define INITGUID

#include "Precomp.h"

EXTERN_C_START

#define QCUDA_MEMORY_TAG ((ULONG)'aduC')
#define RETRY_THRESHOLD 400
#define _4KB 4096
#define _1MB (1024 * 1024)
#define _2MB (2 * _1MB)
#define _4MB (2 * _2MB)
#define Q_PAGE_SIZE _4KB

typedef VirtIODevice VIODEVICE, *PVIODEVICE;
typedef struct virtqueue VIOQUEUE, *PVIOQUEUE;
typedef struct VirtIOBufferDescriptor VIO_SG, *PVIO_SG;

typedef struct va_pa {
	uint64_t physicalMemory;
	uint64_t virtualMemory;
} va_pa, *Pva_pa;

typedef struct _MDL_Ref
{
	PMDL pmdl;
	uint64_t addr;
} AddressMdl, *PAddressMdl;

typedef struct _MemoryBufferEntry
{
	LIST_ENTRY			ListEntry;
	PMDL				PagePmdl;
	PVOID				UvaStart; // start Mapped virtual address
	PVOID				UvaEnd; // end of Mapped virtual
	PAddressMdl			PhysicalAddressMdl; // array of PA pages
	uint64_t			*PhysicalAddress; // base address of PA
	uint64_t			MemorySize; // length allocated
	int					file; // mapping for HostRegister
	uint64_t			data; // pointer for mmap cudaHostGetDevicePointer
} MEMORY_BUFFER_ENTRY, *PMEMORY_BUFFER_ENTRY;

typedef struct _DEVICE_CONTEXT {
	HANDLE   FileHandle; // Store your control data here

	VirtIODevice				VirtDevice;
	struct virtqueue			*VirtQueue;
	PKEVENT						evLowMem;

	// HW Resources
	PVOID						IoBaseAddress;
	ULONG						IoRange;
	BOOLEAN						MappedPort;

	WDFINTERRUPT				WdfInterrupt;
	WDFSPINLOCK					VirtQueueLock;
	WDFSPINLOCK					lock;

	NPAGED_LOOKASIDE_LIST		LookAsideList;
	BOOLEAN						bListInitialized;
	LIST_ENTRY					PageListHead;

} DEVICE_CONTEXT, *PDEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEVICE_CONTEXT, GetDeviceContext)

typedef struct _REQUEST_CONTEXT {
	WDFMEMORY InputMemoryBuffer;
	WDFMEMORY OutputMemoryBuffer;
} REQUEST_CONTEXT, *PREQUEST_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(REQUEST_CONTEXT, GetRequestContext)

DRIVER_INITIALIZE DriverEntry;

EVT_WDF_DRIVER_UNLOAD QcudaEvtDriverUnload;
NTSTATUS QcudaDeviceAdd(IN WDFDRIVER Driver, IN PWDFDEVICE_INIT DeviceInit);
EVT_WDF_DEVICE_CONTEXT_CLEANUP QcudaEvtDriverContextCleanup;
EVT_WDF_DEVICE_CONTEXT_CLEANUP QcudaEvtDeviceContextCleanup;
EVT_WDF_DEVICE_PREPARE_HARDWARE QcudaEvtDevicePrepareHardware;
EVT_WDF_DEVICE_RELEASE_HARDWARE QcudaEvtDeviceReleaseHardware;
EVT_WDF_DEVICE_D0_ENTRY QcudaEvtDeviceD0Entry;
EVT_WDF_DEVICE_D0_EXIT QcudaEvtDeviceD0Exit;
EVT_WDF_IO_QUEUE_IO_DEVICE_CONTROL QcudaEvtIoDeviceControl;

PMEMORY_BUFFER_ENTRY copy_user_to_k(WDFOBJECT WdfDevice, uint64_t from, uint32_t size);
PVIOQUEUE FindVirtualQueue(VIODEVICE *dev, ULONG index);
int QcudaTellHost(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
PMEMORY_BUFFER_ENTRY CreateAndMapMemory(WDFOBJECT WdfDevice, VirtioQCArg **virtioInputArg);
NTSTATUS CreateAndMapMemoryHelper(WDFOBJECT WdfDevice, PMDL* PMemMdl, PVOID* UserVa, uint64_t *size);
NTSTATUS freeMemory(WDFOBJECT WdfDevice, PVOID virtioArg);
uint64_t getGpaOfExistingAddress(WDFOBJECT WdfDevice, PVOID memory);
//uint64_t getVuaOfExistingGpa(WDFOBJECT WdfDevice, PVOID memory);
PMEMORY_BUFFER_ENTRY getMemoryBufferOfExistingAddress(WDFOBJECT WdfDevice, PVOID memory);
PAddressMdl getGvaArray(PVOID memory, size_t length);
void freeGvaArray(PAddressMdl addr);
int mmapctl(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY priv);
void qcummap(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY priv);
int free_host_page_blocks(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY group);
int remove_host_map_file(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY group);
NTSTATUS recordMemoryAllocation(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS releaseMemory(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

// cuda
////////////////////////////////////////////////////////////////////////////////
///	Module & Execution control
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaRegisterFatBinary(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaUnregisterFatBinary(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaRegisterFunction(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaLaunch(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

////////////////////////////////////////////////////////////////////////////////
///	Memory Management
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaMalloc(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaMemset(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaFree(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaMemcpy(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaMemcpyAsync(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

////////////////////////////////////////////////////////////////////////////////
///	Device Management
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaGetDevice(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaGetDeviceCount(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaSetDevice(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaGetDeviceProperties(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaDeviceSynchronize(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaDeviceReset(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

////////////////////////////////////////////////////////////////////////////////
///	Version Management
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaDriverGetVersion(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaRuntimeGetVersion(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

////////////////////////////////////////////////////////////////////////////////
///	Event Management
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaEventCreate(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaEventCreateWithFlags(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaEventRecord(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaEventSynchronize(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaEventElapsedTime(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaEventDestroy(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

////////////////////////////////////////////////////////////////////////////////
/// Error Handling
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaGetLastError(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

////////////////////////////////////////////////////////////////////////////////
///	basic function
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_misc_write(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
int qcu_misc_read(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

///////////////////////////////////////////////////////////////////////////////
///	zero copy
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaHostRegister(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaHostGetDevicePointer(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaHostUnregister(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaSetDeviceFlags(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaFreeHost(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaStreamCreate(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);
NTSTATUS qcu_cudaStreamDestroy(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg);

__inline IsLowMemory(IN WDFOBJECT WdfDevice)
{
	LARGE_INTEGER       TimeOut = { 0 };
	PDEVICE_CONTEXT     devCtx = GetDeviceContext(WdfDevice);

	if (devCtx->evLowMem)
	{
		return (STATUS_WAIT_0 == KeWaitForSingleObject(
			devCtx->evLowMem,
			Executive,
			KernelMode,
			FALSE,
			&TimeOut));
	}
	return FALSE;
}
EXTERN_C_END
