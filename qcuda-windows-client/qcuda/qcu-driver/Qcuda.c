#include "Precomp.h"

#if defined(EVENT_TRACING)
#include "utils.tmh"
#endif

////////////////////////////////////////////////////////////////////////////////
///	Module & Execution control
////////////////////////////////////////////////////////////////////////////////

NTSTATUS qcu_cudaRegisterFatBinary(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status = QcudaTellHost(WdfDevice, virtioArg);

	virtioArg->cmd = VIRTQC_CMD_OPEN;
	virtioArg->pASize = Q_PAGE_SIZE;
	status |= QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaUnregisterFatBinary(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	PDEVICE_CONTEXT context = GetDeviceContext(WdfDevice);
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaRegisterFunction(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	virtioArg->pA = copy_user_to_k(WdfDevice, virtioArg->pA, virtioArg->pASize)->PhysicalAddress;
	virtioArg->pB = copy_user_to_k(WdfDevice, virtioArg->pB, virtioArg->pBSize)->PhysicalAddress;
	status = QcudaTellHost(WdfDevice, virtioArg);
	//freeMemory(WdfDevice, virtioArg->pB); // need to save in host first
	//freeMemory(WdfDevice, virtioArg->pA); // otherwise device reset will crash
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaLaunch(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	
	virtioArg->pA = copy_user_to_k(WdfDevice, virtioArg->pA, virtioArg->pASize)->PhysicalAddress; // (UINT64)physA.QuadPart;
	virtioArg->pB = copy_user_to_k(WdfDevice, virtioArg->pB, virtioArg->pBSize)->PhysicalAddress; // (UINT64)physB.QuadPart;
	status = QcudaTellHost(WdfDevice, virtioArg);
	freeMemory(WdfDevice, virtioArg->pB);
	freeMemory(WdfDevice, virtioArg->pA);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///	Memory Management
////////////////////////////////////////////////////////////////////////////////

NTSTATUS qcu_cudaMalloc(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaMemset(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaFree(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaMemcpy(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	PMEMORY_BUFFER_ENTRY tmp;

	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	virtioArg->rnd = Q_PAGE_SIZE; // *1024;

	if (virtioArg->flag == 1) // cudaMemcpyHostToDevice
	{
		tmp = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pB);
		if (tmp)
		{
			if (tmp->file != -1 && virtioArg->pASize == 0)
			{
				virtioArg->pB = MmGetPhysicalAddress(tmp->PhysicalAddressMdl->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
			}
			else
			{
				virtioArg->pASize = virtioArg->pB - (uint64_t)(tmp->UvaStart);
				uint64_t start = (virtioArg->pASize / Q_PAGE_SIZE + virtioArg->pASize % Q_PAGE_SIZE) * Q_PAGE_SIZE;
				uint64_t addr = virtioArg->pB; // (uint64_t)(tmp->UvaStart) + start; //

				//uint64_t addr = (uint64_t)(virtioArg->pB) + (uint64_t)(virtioArg->pASize);
				PAddressMdl a = getGvaArray(addr, virtioArg->pBSize);
				virtioArg->pB = MmGetPhysicalAddress(a->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
				freeGvaArray(a);//MmFreeContiguousMemory(a);
			}
		}
		else // static address need to allocate memory & copy
		{
			virtioArg->para = 1;
			uint64_t from = virtioArg->pB;
			PMEMORY_BUFFER_ENTRY mem = copy_user_to_k(WdfDevice, from, virtioArg->pBSize);
			virtioArg->pB = mem->PhysicalAddress;
			status = QcudaTellHost(WdfDevice, virtioArg);
			RtlCopyMemory(from, mem->UvaStart, virtioArg->pBSize);
			freeMemory(WdfDevice, mem->UvaStart);
		}
	}
	else if (virtioArg->flag == 2) // cudaMemcpyDeviceToHost
	{
		tmp = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pA);
		if (tmp)
		{
			if (tmp->file != -1 && virtioArg->pBSize == 0)
			{
				virtioArg->pA = MmGetPhysicalAddress(tmp->PhysicalAddressMdl->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
			}
			else {
				virtioArg->pBSize = virtioArg->pA - (uint64_t)(tmp->UvaStart);
				uint64_t start = (virtioArg->pBSize / Q_PAGE_SIZE + virtioArg->pBSize % Q_PAGE_SIZE) * Q_PAGE_SIZE;
				uint64_t addr = virtioArg->pA; // (uint64_t)(tmp->UvaStart) + start; //

				// uint64_t addr = (uint64_t)(virtioArg->pA) + (uint64_t)(virtioArg->pBSize);
				PAddressMdl a = getGvaArray(addr, virtioArg->pASize);
				virtioArg->pA = MmGetPhysicalAddress(a->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
				freeGvaArray(a); //MmFreeContiguousMemory(a);
			}
		}
		else // static address need to allocate memory & copy
		{
			virtioArg->para = 1;
			uint64_t from = virtioArg->pA;
			PMEMORY_BUFFER_ENTRY mem = copy_user_to_k(WdfDevice, from, virtioArg->pASize);
			virtioArg->pA = mem->PhysicalAddress;
			status = QcudaTellHost(WdfDevice, virtioArg);
			RtlCopyMemory(from, mem->UvaStart, virtioArg->pASize);
			freeMemory(WdfDevice, mem->UvaStart);
		}
	}
	else if (virtioArg->flag == 3) // cudaMemcpyDeviceToDevice
	{
		status = QcudaTellHost(WdfDevice, virtioArg);
	}
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaMemcpyAsync(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status = 1;

	if (virtioArg->flag == 1) // cudaMemcpyHostToDevice
	{
		PMEMORY_BUFFER_ENTRY tmp = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pB);
		if (tmp)
		{
			/*if (tmp->MemorySize <= QCU_KMALLOC_MAX_SIZE)
			tmp->PhysicalAddressMdl = getGvaArray(tmp->UvaStart, tmp->MemorySize);*/
			if (tmp->file != -1 && virtioArg->pASize == 0)
			{
				virtioArg->pB = MmGetPhysicalAddress(tmp->PhysicalAddressMdl->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
			}
			else
			{
				virtioArg->pASize = virtioArg->pB - (uint64_t)(tmp->UvaStart);
				uint64_t start = (virtioArg->pASize / Q_PAGE_SIZE + virtioArg->pASize % Q_PAGE_SIZE) * Q_PAGE_SIZE;
				uint64_t addr = virtioArg->pB; //(uint64_t)(tmp->UvaStart) + start; //(uint64_t)(virtioArg->pASize);

				PAddressMdl arr = getGvaArray(addr, virtioArg->pBSize);

				virtioArg->pB = MmGetPhysicalAddress(arr->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
				freeGvaArray(arr);
			}
		}
	}
	else if (virtioArg->flag == 2) // cudaMemcpyDeviceToHost
	{
		PMEMORY_BUFFER_ENTRY tmp = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pA);
		if (tmp)
		{
			/*if (tmp->MemorySize <= QCU_KMALLOC_MAX_SIZE)
			tmp->PhysicalAddressMdl = getGvaArray(tmp->UvaStart, tmp->MemorySize);*/
			if (tmp->file != -1 && virtioArg->pASize == 0)
			{
				virtioArg->pA = MmGetPhysicalAddress(tmp->PhysicalAddressMdl->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
			}
			else
			{
				virtioArg->pBSize = virtioArg->pA - (uint64_t)(tmp->UvaStart);
				uint64_t start = ((virtioArg->pBSize / Q_PAGE_SIZE) + (virtioArg->pBSize % Q_PAGE_SIZE)) * Q_PAGE_SIZE;
				uint64_t addr = virtioArg->pA;// (uint64_t)(tmp->UvaStart) + start; // /*(uint64_t)(virtioArg->pBSize)*/;

				PAddressMdl arr = getGvaArray(addr, virtioArg->pASize);

				virtioArg->pA = MmGetPhysicalAddress(arr->addr).QuadPart;
				status = QcudaTellHost(WdfDevice, virtioArg);
				freeGvaArray(arr);
			}
		}
	}
	else if (virtioArg->flag == 3) // cudaMemcpyDeviceToDevice
	{
		//arg->pA is device pointer
		//arg->pB is device pointer
		status = QcudaTellHost(WdfDevice, virtioArg);
	}
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///	Device Management
////////////////////////////////////////////////////////////////////////////////

NTSTATUS qcu_cudaGetDevice(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaGetDeviceCount(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaSetDevice(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaGetDeviceProperties(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	void *prop = virtioArg->pA;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	PMEMORY_BUFFER_ENTRY t;
	size_t pASize = virtioArg->pASize;
	t = CreateAndMapMemory(WdfDevice, &virtioArg);
	virtioArg->pA = t->PhysicalAddress; // assuming less than one page
	virtioArg->pASize = pASize;
	status = QcudaTellHost(WdfDevice, virtioArg);
	RtlCopyMemory(prop, t->UvaStart, pASize);
	freeMemory(WdfDevice, t->UvaStart);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaDeviceSynchronize(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaDeviceReset(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///	Version Management
////////////////////////////////////////////////////////////////////////////////

NTSTATUS qcu_cudaDriverGetVersion(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaRuntimeGetVersion(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///	Event Management
////////////////////////////////////////////////////////////////////////////////

NTSTATUS qcu_cudaEventCreate(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaEventCreateWithFlags(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaEventRecord(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaEventSynchronize(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaEventElapsedTime(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaEventDestroy(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
/// Error Handling
////////////////////////////////////////////////////////////////////////////////

NTSTATUS qcu_cudaGetLastError(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///	basic function
////////////////////////////////////////////////////////////////////////////////

NTSTATUS qcu_misc_write(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	virtioArg->pA = MmGetPhysicalAddress(virtioArg->pA).QuadPart;
	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

int qcu_misc_read(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	int status;
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	virtioArg->pA = MmGetPhysicalAddress(virtioArg->pA).QuadPart;
	status = QcudaTellHost(WdfDevice, virtioArg);
	return status;
}

int mmapctl(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY priv)
{
	PAGED_CODE();
	VirtioQCArg *arg;
	int status;
	arg = ExAllocatePoolWithTag(
		NonPagedPool,
		sizeof(VirtioQCArg),
		QCUDA_MEMORY_TAG
		);

	arg->cmd = VIRTQC_CMD_MMAPCTL;
	arg->pB = priv->UvaStart;
	arg->pBSize = priv->MemorySize;

	status = QcudaTellHost(WdfDevice, arg);
	priv->file = arg->pA;

	if ((int)(arg->pA) == -1)
		goto err_open;

	return 0;
err_open:
	ExFreePoolWithTag(arg, QCUDA_MEMORY_TAG);
	return -1; //error opening file
}

void qcummap(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY priv)
{
	PAGED_CODE();
	VirtioQCArg *arg;
	PAddressMdl gva_array;
	uint64_t mem;
	int status;

	arg = ExAllocatePoolWithTag(
		NonPagedPool,
		sizeof(VirtioQCArg),
		QCUDA_MEMORY_TAG
		);

	arg->cmd = VIRTQC_CMD_MMAP;
	arg->pA = priv->file;
	arg->pASize = priv->MemorySize / Q_PAGE_SIZE;

	// if it's more we calculate this
	//if (priv->MemorySize <= QCU_KMALLOC_MAX_SIZE)
	priv->PhysicalAddressMdl = getGvaArray(priv->UvaStart, priv->MemorySize);

	arg->pB = MmGetPhysicalAddress(priv->PhysicalAddressMdl->addr).QuadPart;
	arg->rnd = Q_PAGE_SIZE;

	status = QcudaTellHost(WdfDevice, arg);
	priv->file = arg->pA;

	/*MmUnmapLockedPages(gva_array->ptr, gva_array->pmdl);
	MmFreePagesFromMdl(gva_array->pmdl);
	ExFreePool(gva_array->pmdl);
	ExFreePoolWithTag(gva_array->pmdl, QCUDA_MEMORY_TAG);*/
	ExFreePoolWithTag(arg, QCUDA_MEMORY_TAG);
}

///////////////////////////////////////////////////////////////////////////////
///	zero copy
////////////////////////////////////////////////////////////////////////////////
NTSTATUS qcu_cudaHostRegister(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	int status;
	void *pA = virtioArg->pA;
	size_t size = virtioArg->pASize;

	PMEMORY_BUFFER_ENTRY tmp = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pA);

	mmapctl(WdfDevice, tmp);
	qcummap(WdfDevice, tmp);

	virtioArg->pB = tmp->MemorySize;
	virtioArg->pBSize = tmp->file;
	virtioArg->rnd = (uint64_t)(virtioArg->pA) - (uint64_t)(tmp->UvaStart); // offset

	virtioArg->pA = MmGetPhysicalAddress(tmp->PhysicalAddressMdl->addr).QuadPart;
	status = QcudaTellHost(WdfDevice, virtioArg);
	tmp->data = virtioArg->rnd;

	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaHostGetDevicePointer(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status;

	PMEMORY_BUFFER_ENTRY tmp = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pB);

	virtioArg->pB = tmp->data;

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaHostUnregister(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status;

	PMEMORY_BUFFER_ENTRY tmp = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pA);

	virtioArg->pA = MmGetPhysicalAddress(tmp->PhysicalAddressMdl->addr).QuadPart;
	virtioArg->pASize = tmp->MemorySize;

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaSetDeviceFlags(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status;

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaFreeHost(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status;

	// freeMemory(WdfDevice, virtioArg->pA);
	return releaseMemory(WdfDevice, virtioArg);
}

NTSTATUS qcu_cudaStreamCreate(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status;

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}

NTSTATUS qcu_cudaStreamDestroy(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);
	int status;

	status = QcudaTellHost(WdfDevice, virtioArg);
	return (status) ? STATUS_UNSUCCESSFUL : STATUS_SUCCESS;
}



