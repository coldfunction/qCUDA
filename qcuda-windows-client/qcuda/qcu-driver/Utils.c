#include "Precomp.h"

#if defined(EVENT_TRACING)
#include "utils.tmh"
#endif

PMEMORY_BUFFER_ENTRY copy_user_to_k(WDFOBJECT WdfDevice, uint64_t from, uint32_t size)
{
	//uint64_t addr = 0;
	VirtioQCArg *buffer;
	PMEMORY_BUFFER_ENTRY p;
	p = getMemoryBufferOfExistingAddress(WdfDevice, from); // we have it already
	if (!p)
	{
		buffer = ExAllocatePoolWithTag(
			NonPagedPool,
			sizeof(VirtioQCArg),
			QCUDA_MEMORY_TAG
			);

		buffer->pASize = size;
		p = CreateAndMapMemory(WdfDevice, &buffer);

		RtlCopyMemory(buffer->pA, from, size);
		ExFreePoolWithTag(buffer, QCUDA_MEMORY_TAG);
		//addr = p->PhysicalAddress;
	}
	return p;
}

PVIOQUEUE FindVirtualQueue(VIODEVICE *dev, ULONG index)
{
	PAGED_CODE();
	PVIOQUEUE  pq = NULL;
	PVOID p;
	ULONG size, allocSize;
	VirtIODeviceQueryQueueAllocation(dev, index, &size, &allocSize);
	if (allocSize)
	{
		PHYSICAL_ADDRESS HighestAcceptable;
		HighestAcceptable.QuadPart = 0xFFFFFFFFFF;
		p = MmAllocateContiguousMemory(allocSize, HighestAcceptable);
		if (p)
		{
			pq = VirtIODevicePrepareQueue(dev, index, MmGetPhysicalAddress(p), p, allocSize, p, FALSE);
		}
	}
	return pq;
}

PMEMORY_BUFFER_ENTRY CreateAndMapMemory(WDFOBJECT WdfDevice, VirtioQCArg **virtioInputArg)
{
	PAGED_CODE();
	PDEVICE_CONTEXT context = GetDeviceContext(WdfDevice);
	PMEMORY_BUFFER_ENTRY pNewPageListEntry;
	PMDL pPageMdl = NULL;
	NTSTATUS status = STATUS_SUCCESS;
	PVOID UserVa; // = (void*)((*virtioOutputArg)->pA);
	uint64_t size = (*virtioInputArg)->pASize, j = 0;
	uint64_t i = 0;
	
	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_HW_ACCESS, "--> %s\n", __FUNCTION__);

	if (size <= 0)
		return NULL; // STATUS_INVALID_PARAMETER;

	//if (IsLowMemory(WdfDevice))
	//{
	//	/*TraceEvents(TRACE_LEVEL_WARNING, DBG_HW_ACCESS,
	//		"Low memory. Allocated pages: %d\n", context->num_pages);*/
	//	return  NULL; // STATUS_RESOURCE_IN_USE;
	//}

	//context->num_pfns = 0;

	pNewPageListEntry = (PMEMORY_BUFFER_ENTRY)ExAllocateFromNPagedLookasideList( &context->LookAsideList);

	if (pNewPageListEntry == NULL)
	{
		/*TraceEvents(TRACE_LEVEL_ERROR, DBG_HW_ACCESS, "Failed to allocate list entry.\n");*/
		return STATUS_INSUFFICIENT_RESOURCES;
	} else {
		status = CreateAndMapMemoryHelper(WdfDevice, &pPageMdl, &UserVa, &size);
		if (!NT_SUCCESS(status))
		{
			return NULL; // status;
		} else {
			pNewPageListEntry->PagePmdl = pPageMdl;
			pNewPageListEntry->UvaStart = UserVa;
			pNewPageListEntry->UvaEnd = (uint64_t)UserVa + size;
			pNewPageListEntry->MemorySize = size;
			pNewPageListEntry->file = -1;
			//
			pNewPageListEntry->PhysicalAddress = MmGetPhysicalAddress(UserVa).QuadPart;

			InsertHeadList(&context->PageListHead, &pNewPageListEntry->ListEntry);

			(*virtioInputArg)->pA = UserVa;
			(*virtioInputArg)->pASize = sizeof(UserVa);
			(*virtioInputArg)->flag = 1; //
		}
	}
	return pNewPageListEntry;
}

NTSTATUS CreateAndMapMemoryHelper(WDFOBJECT WdfDevice, PMDL* PMemMdl, PVOID* UserVa, uint64_t *size)
{
	PAGED_CODE();
	PMDL                mdl;
	PVOID               userVAToReturn;
	PHYSICAL_ADDRESS    lowAddress;
	PHYSICAL_ADDRESS    highAddress;
	ULONG				allocationFlag = 0;

	int blockSize = Q_PAGE_SIZE; // *1024;
	unsigned int numOfBlocks = (*size/blockSize) + (*size%blockSize > 0);
	// if (*size%blockSize != 0) numOfBlocks++;

	*size = numOfBlocks * blockSize; 
	
	lowAddress.QuadPart = 0;
	highAddress.QuadPart = (ULONGLONG)-1;// 0xFFFFFFFFFFFFFFFF;
	
	if (*size <= QCU_KMALLOC_MAX_SIZE) // 4MB 4194304
		allocationFlag = MM_ALLOCATE_REQUIRE_CONTIGUOUS_CHUNKS;
	
	mdl = MmAllocatePagesForMdlEx(lowAddress, highAddress, lowAddress, *size, MmCached, allocationFlag);
	
	if (!mdl) {
		/*TraceEvents(TRACE_LEVEL_WARNING, DBG_HW_ACCESS,
			"Failed to allocate pages.\n");*/
		return STATUS_INSUFFICIENT_RESOURCES;
	}

	if (MmGetMdlByteCount(mdl) != *size)
	{
		/*TraceEvents(TRACE_LEVEL_WARNING, DBG_HW_ACCESS,
			"Not all requested memory was allocated (%d/%d).\n",
			MmGetMdlByteCount(mdl), size);*/
		MmFreePagesFromMdl(mdl);
		ExFreePool(mdl);
		return STATUS_INSUFFICIENT_RESOURCES;
	}

	userVAToReturn = MmMapLockedPagesSpecifyCache(mdl, // MDL
		UserMode,     // Mode
		MmCached,     // Caching
		NULL,         // Address
		FALSE,        // Bugcheck?
		NormalPagePriority); // Priority

	if (!userVAToReturn) {
		MmFreePagesFromMdl(mdl);
		IoFreeMdl(mdl);
		return STATUS_INSUFFICIENT_RESOURCES;
	}

	*UserVa = userVAToReturn;
	*PMemMdl = mdl;

	return STATUS_SUCCESS;
}

NTSTATUS freeMemory(WDFOBJECT WdfDevice, PVOID memory)
{
	PDEVICE_CONTEXT context = GetDeviceContext(WdfDevice);
	PMEMORY_BUFFER_ENTRY temp;
	PMDL pPageMdl;
	// size_t pages = 0;
	PLIST_ENTRY iter = NULL;
	NTSTATUS status = STATUS_UNSUCCESSFUL;
	
	iter = context->PageListHead.Flink;

	WdfSpinLockAcquire(context->lock);
	while (iter != &(context->PageListHead))
	{
		PMEMORY_BUFFER_ENTRY temp = (PMEMORY_BUFFER_ENTRY)CONTAINING_RECORD(iter, MEMORY_BUFFER_ENTRY, ListEntry);

		if (temp->UvaStart == memory || temp->PhysicalAddress == memory) // found a match
		{
			pPageMdl = temp->PagePmdl;
			
			if (/*temp->PhysicalAddressMdl != NULL ||*/ temp->file != -1) //>MemorySize > QCU_KMALLOC_MAX_SIZE
			{
				free_host_page_blocks(WdfDevice, temp);
				remove_host_map_file(WdfDevice, temp);
				freeGvaArray(temp->PhysicalAddressMdl);
			}

			RemoveEntryList(&(temp->ListEntry));
			MmUnmapLockedPages(temp->UvaStart, pPageMdl);
			MmFreePagesFromMdl(pPageMdl);
			ExFreePool(pPageMdl);
			ExFreeToNPagedLookasideList(&context->LookAsideList, temp);
			
			status = STATUS_SUCCESS;
			break;
		}

		iter = iter->Flink;
	}
	WdfSpinLockRelease(context->lock);

	return status;
}

uint64_t getGpaOfExistingAddress(WDFOBJECT WdfDevice, PVOID memory)
{
	PAGED_CODE();
	PDEVICE_CONTEXT context = GetDeviceContext(WdfDevice);
	PMEMORY_BUFFER_ENTRY temp;
	PLIST_ENTRY iter = NULL;
	NTSTATUS status = STATUS_UNSUCCESSFUL;

	iter = context->PageListHead.Flink;
	temp = getMemoryBufferOfExistingAddress(WdfDevice, memory);

	if (temp)
	{
		if (temp->MemorySize > QCU_KMALLOC_MAX_SIZE) // not contiguous
			return (uint64_t)-1;
		return MmGetPhysicalAddress(temp->UvaStart).QuadPart;
	}

	return (uint64_t)-1;
}

PMEMORY_BUFFER_ENTRY getMemoryBufferOfExistingAddress(WDFOBJECT WdfDevice, PVOID memory)
{
	PAGED_CODE();
	PDEVICE_CONTEXT context = GetDeviceContext(WdfDevice);
	// PMEMORY_BUFFER_ENTRY temp;
	PLIST_ENTRY iter = NULL;
	NTSTATUS status = STATUS_UNSUCCESSFUL;

	iter = context->PageListHead.Flink;
	while (iter != &(context->PageListHead))
	{
		PMEMORY_BUFFER_ENTRY temp = (PMEMORY_BUFFER_ENTRY)CONTAINING_RECORD(iter, MEMORY_BUFFER_ENTRY, ListEntry);

		if (temp->UvaStart <= memory && temp->UvaEnd > memory) // found a match
			return temp;

		iter = iter->Flink;
	}
	return NULL;
}

PAddressMdl getGvaArray(PVOID memory, size_t length)
{
	PAGED_CODE();
	PMDL                mdl;
	PAddressMdl			ret;
	PHYSICAL_ADDRESS    lowAddress;
	PHYSICAL_ADDRESS    highAddress;
	uint64_t *gpaArray, i = 0;
	uint64_t endOfAddress = (uint64_t)memory + (uint64_t)length;
	int j = 0;
	
	lowAddress.QuadPart = 0;
	highAddress.QuadPart = (ULONGLONG)-1;

	ret = ExAllocatePoolWithTag(
		NonPagedPool,
		sizeof(PAddressMdl),
		QCUDA_MEMORY_TAG
		);

	mdl = MmAllocatePagesForMdlEx(lowAddress, 
		highAddress, 
		lowAddress, 
		(length/QCU_KMALLOC_MAX_SIZE)+1, //QCU_KMALLOC_MAX_SIZE, 
		MmCached, MM_ALLOCATE_REQUIRE_CONTIGUOUS_CHUNKS);

	if (!mdl) {
		return NULL;
	}

	gpaArray = MmMapLockedPagesSpecifyCache(mdl, // MDL
		UserMode,     // Mode
		MmCached,     // Caching
		NULL,         // Address
		FALSE,        // Bugcheck?
		NormalPagePriority); // Priority

	if (!gpaArray) {
		MmFreePagesFromMdl(mdl);
		IoFreeMdl(mdl);
		return NULL;
	}

	i = (uint64_t)memory;
	while (i < endOfAddress)
	{
		gpaArray[j] = MmGetPhysicalAddress(i).QuadPart;
		i += (uint64_t)Q_PAGE_SIZE;
		j++;
	}
	ret->pmdl = mdl;
	ret->addr = gpaArray;
	return ret;
}

void freeGvaArray(PAddressMdl addr)
{
	MmUnmapLockedPages(addr->addr, addr->pmdl);
	MmFreePagesFromMdl(addr->pmdl);
	ExFreePool(addr->pmdl);
	ExFreePoolWithTag(addr, QCUDA_MEMORY_TAG);
}

int free_host_page_blocks(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY group)
{
	// PAGED_CODE(); // called from spin lock region
	VirtioQCArg *virtioArg
		= ExAllocatePoolWithTag(
			NonPagedPool,
			sizeof(VirtioQCArg),
			QCUDA_MEMORY_TAG
			);

	unsigned int numOfblocks = group->MemorySize/Q_PAGE_SIZE;
	virtioArg->pB = MmGetPhysicalAddress(group->PhysicalAddressMdl->addr).QuadPart;
	virtioArg->pBSize = numOfblocks;
	virtioArg->cmd = VIRTQC_CMD_MUNMAP;

	QcudaTellHost(WdfDevice, virtioArg);
	
	ExFreePoolWithTag(virtioArg, QCUDA_MEMORY_TAG);
	return 0;
}

int remove_host_map_file(WDFOBJECT WdfDevice, PMEMORY_BUFFER_ENTRY group)
{
	// PAGED_CODE(); // called from spin lock region
	VirtioQCArg *virtioArg
		= ExAllocatePoolWithTag(
			NonPagedPool,
			sizeof(VirtioQCArg),
			QCUDA_MEMORY_TAG
			);

	virtioArg->pA = group->UvaStart;
	virtioArg->pBSize = group->file;
	virtioArg->cmd = VIRTQC_CMD_MMAPRELEASE;
	
	QcudaTellHost(WdfDevice, virtioArg);

	ExFreePoolWithTag(virtioArg, QCUDA_MEMORY_TAG);
	return 0;
}

int QcudaTellHost(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	VIO_SG              sg[2];
	PDEVICE_CONTEXT     context = GetDeviceContext(WdfDevice);
	unsigned int		len = 0;
	int					cnt = 0, error = 0;
	void				*data = NULL;

	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "--> %s\n", __FUNCTION__);

	sg[0].physAddr = sg[1].physAddr = MmGetPhysicalAddress(virtioArg);
	sg[0].length = sg[1].length = sizeof(VirtioQCArg);

	WdfSpinLockAcquire(context->VirtQueueLock);

	error = virtqueue_add_buf(context->VirtQueue, sg, 1, 1, virtioArg, NULL, 0);
	if (error == 0)
	{
		virtqueue_kick(context->VirtQueue);
		while (!data)
		{
			data = virtqueue_get_buf(context->VirtQueue, &len);
			KeStallExecutionProcessor(10); /// busy waiting cpu_relax like
			if (++cnt > RETRY_THRESHOLD)
			{
				//TraceEvents(TRACE_LEVEL_FATAL, DBG_PNP, "<-> %s retries = %d\n", __FUNCTION__, cnt);
				break;
			}
		}
	}
	
	WdfSpinLockRelease(context->VirtQueueLock);

	RtlCopyMemory(virtioArg, data, sizeof(VirtioQCArg));
	return error;
}

NTSTATUS recordMemoryAllocation(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	PDEVICE_CONTEXT context = GetDeviceContext(WdfDevice);
	PMEMORY_BUFFER_ENTRY pNewPageListEntry;
	NTSTATUS status = STATUS_SUCCESS;
	uint64_t i = 0;

	pNewPageListEntry = (PMEMORY_BUFFER_ENTRY)ExAllocateFromNPagedLookasideList(&context->LookAsideList);

	if (pNewPageListEntry)
	{
		pNewPageListEntry->UvaStart = virtioArg->pA;
		pNewPageListEntry->UvaEnd = (uint64_t)virtioArg->pA + virtioArg->pASize;
		pNewPageListEntry->MemorySize = virtioArg->pASize;
		pNewPageListEntry->file = -1;

		// no PMDL assign should never call free on this node
		InsertHeadList(&context->PageListHead, &pNewPageListEntry->ListEntry);	
	}
	else 
		status = STATUS_UNSUCCESSFUL;

	return status;
}

NTSTATUS releaseMemory(WDFOBJECT WdfDevice, VirtioQCArg *virtioArg)
{
	PAGED_CODE();
	PDEVICE_CONTEXT context = GetDeviceContext(WdfDevice);
	PMEMORY_BUFFER_ENTRY buffer;
	NTSTATUS status = STATUS_SUCCESS;
	
	buffer = getMemoryBufferOfExistingAddress(WdfDevice, virtioArg->pA);
	if (buffer)
	{
		// PMEMORY_BUFFER_ENTRY temp = (PMEMORY_BUFFER_ENTRY)CONTAINING_RECORD(iter, MEMORY_BUFFER_ENTRY, ListEntry);
		virtioArg->pA = buffer->UvaStart;
		virtioArg->pASize = buffer->MemorySize;
		virtioArg->flag = 1;

		if (buffer->file != -1)
		{
			free_host_page_blocks(WdfDevice, buffer);
			remove_host_map_file(WdfDevice, buffer);
			freeGvaArray(buffer->PhysicalAddressMdl);
		}

		RemoveEntryList(&(buffer->ListEntry));
		ExFreeToNPagedLookasideList(&context->LookAsideList, buffer);
	}
	else
		virtioArg->flag = 0;
}