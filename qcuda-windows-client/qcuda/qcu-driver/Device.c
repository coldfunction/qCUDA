#include "Precomp.h"

#if defined(EVENT_TRACING)
#include "device.tmh"
#endif

#ifdef ALLOC_PRAGMA
#pragma alloc_text( PAGE, QcudaEvtDeviceContextCleanup)
#pragma alloc_text( PAGE, QcudaDeviceAdd)
#pragma alloc_text( PAGE, QcudaEvtDevicePrepareHardware)
#pragma alloc_text( PAGE, QcudaEvtDeviceReleaseHardware)
#pragma alloc_text( PAGE, QcudaEvtDeviceD0Entry)
#pragma alloc_text( PAGE, QcudaEvtDeviceD0Exit)
#pragma alloc_text( PAGE, QcudaEvtIoDeviceControl)
//#pragma alloc_text( PAGE, QcudaEvtFileClose)
#endif

NTSTATUS
QcudaDeviceAdd(
	IN WDFDRIVER  Driver,
	IN PWDFDEVICE_INIT  DeviceInit)
{
	NTSTATUS                     status = STATUS_SUCCESS;
	WDFDEVICE                    device;
	PDEVICE_CONTEXT				 context;
	WDF_INTERRUPT_CONFIG         interruptConfig;
	WDF_OBJECT_ATTRIBUTES        attributes;
	WDF_PNPPOWER_EVENT_CALLBACKS pnpPowerCallbacks;
	WDF_FILEOBJECT_CONFIG		 fileConfig;
	WDF_IO_QUEUE_CONFIG			 queueConfig;
	WDFQUEUE                     queue;

	DECLARE_CONST_UNICODE_STRING(ntDeviceName, DEV_NAME);
	DECLARE_CONST_UNICODE_STRING(symbolicLinkName, DEV_SYMBOLIC_LINK);

	UNREFERENCED_PARAMETER(Driver);
	PAGED_CODE();

	// TraceEvents(TRACE_LEVEL_INFORMATION, DBG_PNP, "--> %s\n", __FUNCTION__);
	
	WdfDeviceInitSetExclusive(DeviceInit, TRUE);

	status = WdfDeviceInitAssignName(DeviceInit, &ntDeviceName);

	if (!NT_SUCCESS(status)) {
		// TraceEvents(TRACE_LEVEL_ERROR, DBG_PNP, "WdfDeviceInitAssignName failed %!STATUS!", status);
		goto End;
	}
	WdfDeviceInitSetDeviceClass(DeviceInit, &GUID_DEVINTERFACE_qcudriver);

	WdfDeviceInitAssignSDDLString(DeviceInit, &SDDL_DEVOBJ_SYS_ALL_ADM_RWX_WORLD_RWX_RES_RWX); // dangerous :|

	//WDF_FILEOBJECT_CONFIG_INIT(
	//	&fileConfig,
	//	QcudaEvtDeviceFileCreate,
	//	QcudaEvtFileClose,
	//	WDF_NO_EVENT_CALLBACK // not interested in Cleanup
	//	);

	//WdfDeviceInitSetFileObjectConfig(DeviceInit,
	//	&fileConfig,
	//	WDF_NO_OBJECT_ATTRIBUTES);

	WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&pnpPowerCallbacks);
	pnpPowerCallbacks.EvtDevicePrepareHardware = QcudaEvtDevicePrepareHardware;
	pnpPowerCallbacks.EvtDeviceReleaseHardware = QcudaEvtDeviceReleaseHardware;
	pnpPowerCallbacks.EvtDeviceD0Entry = QcudaEvtDeviceD0Entry;

	WdfDeviceInitSetPnpPowerEventCallbacks(DeviceInit, &pnpPowerCallbacks);
	WdfDeviceInitSetIoType(DeviceInit, WdfDeviceIoBuffered);
	
	WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&attributes, DEVICE_CONTEXT);
	attributes.EvtCleanupCallback = QcudaEvtDeviceContextCleanup;

	status = WdfDeviceCreate(&DeviceInit, &attributes, &device);

	if (!NT_SUCCESS(status))
	{
		/*TraceEvents(TRACE_LEVEL_ERROR, DBG_PNP,
			"WdfDeviceCreate failed with status 0x%08x\n", status);*/
		return status;
	}

	WdfDeviceCreateSymbolicLink(device, &symbolicLinkName);

	context = GetDeviceContext(device);

	WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
	attributes.ParentObject = device;
	status = WdfSpinLockCreate(&attributes, &context->VirtQueueLock);
	status |= WdfSpinLockCreate(&attributes, &context->lock);

	VirtIODeviceReset(&context->VirtDevice);

	if (!NT_SUCCESS(status))
	{
		/*TraceEvents(TRACE_LEVEL_ERROR, DBG_INIT,
			"WdfSpinLockCreate failed: %!STATUS!", status);*/
		return status;
	}

	status = WdfDeviceCreateDeviceInterface(device,
		&GUID_DEVINTERFACE_qcudriver, NULL);

	if (!NT_SUCCESS(status))
	{
		/*TraceEvents(TRACE_LEVEL_ERROR, DBG_INIT,
			"WdfDeviceCreateDeviceInterface failed: %!STATUS!", status);*/
		return status;
	}

	//context->num_pages = 0;
	// context->PageListHead.Next = NULL;
	InitializeListHead(&(context->PageListHead));

	ExInitializeNPagedLookasideList(
		&context->LookAsideList,
		NULL,
		NULL,
		0,
		sizeof(MEMORY_BUFFER_ENTRY),
		QCUDA_MEMORY_TAG,
		0
		);

	context->bListInitialized = TRUE;
	
	WDF_IO_QUEUE_CONFIG_INIT(&queueConfig, WdfIoQueueDispatchSequential);

	queueConfig.AllowZeroLengthRequests = TRUE;
	queueConfig.EvtIoDeviceControl = QcudaEvtIoDeviceControl;

	WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
	__analysis_assume(queueConfig.EvtIoStop != 0);
	status = WdfIoQueueCreate(device,
		&queueConfig,
		&attributes,
		&queue // pointer to default queue
		);
	__analysis_assume(queueConfig.EvtIoStop == 0);
	if (!NT_SUCCESS(status)) {
		// TraceEvents(TRACE_LEVEL_ERROR, DBG_INIT, "WdfIoQueueCreate failed %!STATUS!", status);
		goto End;
	}

	status |= WdfDeviceConfigureRequestDispatching(device,
		queue, WdfRequestTypeDeviceControl);

	if (!NT_SUCCESS(status))
	{
		/*TraceEvents(TRACE_LEVEL_ERROR, DBG_INIT,
			"WdfDeviceConfigureRequestDispatching failed: %!STATUS!", status);*/
		return status;
	}

	WdfControlFinishInitializing(device);

End:
	if (DeviceInit != NULL) {
		WdfDeviceInitFree(DeviceInit);
	}

	return status;
}

NTSTATUS
QcudaEvtDevicePrepareHardware(
	IN WDFDEVICE    Device,
	IN WDFCMRESLIST ResourceList,
	IN WDFCMRESLIST ResourceListTranslated
	)
{
	PDEVICE_CONTEXT context = GetDeviceContext(Device);
	PCM_PARTIAL_RESOURCE_DESCRIPTOR desc;
	BOOLEAN signaled = FALSE;
	ULONG i;

	UNREFERENCED_PARAMETER(ResourceList);

	/*TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER, "--> %!FUNC! Device: %p",
		Device);*/

	PAGED_CODE();

	for (i = 0; i < WdfCmResourceListGetCount(ResourceListTranslated); ++i)
	{
		desc = WdfCmResourceListGetDescriptor(ResourceListTranslated, i);
		switch (desc->Type)
		{
		case CmResourceTypePort:
		{
			/*TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER,
				"I/O mapped CSR: (%x) Length: (%d)",
				desc->u.Port.Start.LowPart, desc->u.Port.Length);*/

			context->MappedPort = !(desc->Flags & CM_RESOURCE_PORT_IO);
			context->IoRange = desc->u.Port.Length;

			if (context->MappedPort)
			{
				context->IoBaseAddress = MmMapIoSpace(desc->u.Port.Start,
					desc->u.Port.Length, MmNonCached);
			}
			else
			{
				context->IoBaseAddress =
					(PVOID)(ULONG_PTR)desc->u.Port.Start.QuadPart;
			}

			break;
		}

		case CmResourceTypeInterrupt:
		{
			signaled = !!(desc->Flags &
				(CM_RESOURCE_INTERRUPT_LATCHED | CM_RESOURCE_INTERRUPT_MESSAGE));

			/*TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER,
				"Interrupt Level: 0x%08x, Vector: 0x%08x Signaled: %!BOOLEAN!",
				desc->u.Interrupt.Level, desc->u.Interrupt.Vector, signaled);*/

			break;
		}

		default:
			break;
		}
	}

	if (!context->IoBaseAddress)
	{
		//TraceEvents(TRACE_LEVEL_ERROR, DBG_POWER, "Port not found.");
		return STATUS_INSUFFICIENT_RESOURCES;
	}

	VirtIODeviceInitialize(&context->VirtDevice,
		(ULONG_PTR)(context->IoBaseAddress), sizeof(context->VirtDevice)); // VirtIODeviceSizeRequired(1)
	VirtIODeviceSetMSIXUsed(&context->VirtDevice, signaled);
	VirtIODeviceReset(&context->VirtDevice);

	if (signaled)
	{
		WriteVirtIODeviceWord(
			context->VirtDevice.addr + VIRTIO_MSI_CONFIG_VECTOR, 1);
		(VOID)ReadVirtIODeviceWord(
			context->VirtDevice.addr + VIRTIO_MSI_CONFIG_VECTOR);
	}

	VirtIODeviceAddStatus(&context->VirtDevice, VIRTIO_CONFIG_S_ACKNOWLEDGE);

	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER, "<-- %!FUNC!");

	return STATUS_SUCCESS;
}

NTSTATUS
QcudaEvtDeviceReleaseHardware(
	IN WDFDEVICE      Device,
	IN WDFCMRESLIST   ResourcesTranslated
	)
{
	PDEVICE_CONTEXT context = GetDeviceContext(Device);

	UNREFERENCED_PARAMETER(ResourcesTranslated);

	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER, "--> %!FUNC!");

	PAGED_CODE();

	if (context->MappedPort && context->IoBaseAddress)
	{
		MmUnmapIoSpace(context->IoBaseAddress, context->IoRange);
		context->IoBaseAddress = NULL;
	}

	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER, "<-- %!FUNC!");

	return STATUS_SUCCESS;
}

NTSTATUS
QcudaEvtDeviceD0Entry(
	IN  WDFDEVICE Device,
	IN  WDF_POWER_DEVICE_STATE PreviousState
	)
{
	NTSTATUS            status = STATUS_SUCCESS;
	PDEVICE_CONTEXT devCtx = GetDeviceContext(Device);
	DWORD MemorySizeLower = 128 * 1024 * 1024;
	DWORD MemorySizeUpper = 1024 * 1024 * 1024;

	UNREFERENCED_PARAMETER(PreviousState);
	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_INIT, "--> %s\n", __FUNCTION__);
	PDEVICE_CONTEXT context = GetDeviceContext(Device);

	context->VirtQueue = FindVirtualQueue(&context->VirtDevice, 0);
	// SetProcessWorkingSetSize(GetCurrentProcess(), MemorySizeLower, MemorySizeUpper);

	return status;
}

NTSTATUS QcudaEvtDeviceD0Exit(IN WDFDEVICE Device,
	IN WDF_POWER_DEVICE_STATE TargetState)
{
	PDEVICE_CONTEXT context = GetDeviceContext(Device);

	UNREFERENCED_PARAMETER(TargetState);

	/*TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER, "--> %!FUNC! Device: %p",
		Device);*/

	PAGED_CODE();

	VirtIODeviceRemoveStatus(&context->VirtDevice, VIRTIO_CONFIG_S_DRIVER_OK);
	// DeleteQueue(&context->VirtQueue);
	// TODO: memory cleanup

	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER, "<-- %!FUNC!");

	return STATUS_SUCCESS;
}

VOID QcudaEvtDeviceContextCleanup(IN WDFOBJECT DeviceObject)
{
	PDEVICE_CONTEXT context = GetDeviceContext(DeviceObject);
	PSINGLE_LIST_ENTRY iter;

	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_INIT, "--> %!FUNC!");
	WdfSpinLockAcquire(context->VirtQueueLock);
	if (context->bListInitialized)
	{
		ExDeleteNPagedLookasideList(&context->LookAsideList);
		context->bListInitialized = FALSE;
	}
	WdfSpinLockRelease(context->VirtQueueLock);

	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_INIT, "<-- %!FUNC!");
}

VOID
QcudaEvtIoDeviceControl(
	IN WDFQUEUE         Queue,
	IN WDFREQUEST       Request,
	IN size_t            OutputBufferLength,
	IN size_t            InputBufferLength,
	IN ULONG            IoControlCode
	)
{
	NTSTATUS            status = STATUS_SUCCESS;// Assume success
	PREQUEST_CONTEXT    reqContext = NULL;
	size_t              bufferSize;
	WDFDEVICE			device = WdfIoQueueGetDevice(Queue);
	// PDEVICE_CONTEXT		context = GetDeviceContext(device);
	PVOID				inputBuffer = NULL, ioBuffer = NULL;
	VirtioQCArg			*inputVirtArg = NULL, *tempInput, zeroedVirt; // *outputVirtArg = NULL,
	PIRP				irp = WdfRequestWdmGetIrp(Request);
	//PIO_STACK_LOCATION	io = IoGetCurrentIrpStackLocation(irp); // nextStack = IoGetNextIrpStackLocation(irp);

	UNREFERENCED_PARAMETER(Queue);

	PAGED_CODE();

	inputVirtArg = ExAllocatePoolWithTag(
		NonPagedPool,
		sizeof(VirtioQCArg),
		QCUDA_MEMORY_TAG
		);

	memset(&zeroedVirt, 0, sizeof(VirtioQCArg));

	// input and output can be null 
	//if (!OutputBufferLength && !InputBufferLength)
	//{
	//	WdfRequestComplete(Request, STATUS_INVALID_PARAMETER);
	//	return;
	//} 

	//if (InputBufferLength) 
	//{
	status = WdfRequestRetrieveInputBuffer(Request, 0, &tempInput, &InputBufferLength);
	
	if (tempInput)
		RtlCopyMemory(inputVirtArg, tempInput, sizeof(VirtioQCArg));
	else
		RtlCopyMemory(inputVirtArg, &zeroedVirt, sizeof(VirtioQCArg));

	inputVirtArg->cmd = IoControlCode;
	
	ioBuffer = irp->AssociatedIrp.SystemBuffer;

	switch (IoControlCode) // io->Parameters.DeviceIoControl.IoControlCode
	{
	case VIRTQC_CMD_MMAP:
		status = recordMemoryAllocation(device, inputVirtArg);
		break;

	case VIRTQC_CMD_MMAPRELEASE:
		status = releaseMemory(device, inputVirtArg);
		break;

	// Module & Execution control (driver API)
	case VIRTQC_cudaRegisterFatBinary:
		//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_IOCTLS, "fat bin %x\n", IoControlCode);
		status = qcu_cudaRegisterFatBinary(device, inputVirtArg);
		break;
	case VIRTQC_cudaUnregisterFatBinary:
		status = qcu_cudaUnregisterFatBinary(device, inputVirtArg);
		break;

	case VIRTQC_cudaRegisterFunction:
		status = qcu_cudaRegisterFunction(device, inputVirtArg);
		break;

	case VIRTQC_cudaLaunch:
		status = qcu_cudaLaunch(device, inputVirtArg);
		break;

	// Memory Management (runtime API)
	case VIRTQC_cudaMalloc:
		status = qcu_cudaMalloc(device, inputVirtArg);
		break;

	case VIRTQC_cudaMemset:
		status = qcu_cudaMemset(device, inputVirtArg);
		break;

	case VIRTQC_cudaMemcpy:
		status = qcu_cudaMemcpy(device, inputVirtArg);
		break;

	case VIRTQC_cudaMemcpyAsync:
		status = qcu_cudaMemcpyAsync(device, inputVirtArg);
		break;

	case VIRTQC_cudaFree:
		status = qcu_cudaFree(device, inputVirtArg);
		break;

	// Device Management (runtime API)
	case VIRTQC_cudaGetDevice:
		status = qcu_cudaGetDevice(device, inputVirtArg);
		break;

	case VIRTQC_cudaGetDeviceCount:
		status = qcu_cudaGetDeviceCount(device, inputVirtArg);
		break;

	case VIRTQC_cudaSetDevice:
		status = qcu_cudaSetDevice(device, inputVirtArg);
		break;
	case VIRTQC_cudaGetDeviceProperties:
		status = qcu_cudaGetDeviceProperties(device, inputVirtArg);
		break;

	case VIRTQC_cudaDeviceSynchronize:
		status = qcu_cudaDeviceSynchronize(device, inputVirtArg);
		break;

	case VIRTQC_cudaDeviceReset:
		status = qcu_cudaDeviceReset(device, inputVirtArg);
		break;

	// Version Management (runtime API)
	case VIRTQC_cudaDriverGetVersion:
		status = qcu_cudaDriverGetVersion(device, inputVirtArg);
		break;

	case VIRTQC_cudaRuntimeGetVersion:
		status = qcu_cudaRuntimeGetVersion(device, inputVirtArg);
		break;

		// Event Management (runtime API)
	case VIRTQC_cudaEventCreate:
		status = qcu_cudaEventCreate(device, inputVirtArg);
		break;

	case VIRTQC_cudaEventCreateWithFlags:
		status = qcu_cudaEventCreateWithFlags(device, inputVirtArg);
		break;

	case VIRTQC_cudaEventRecord:
		status = qcu_cudaEventRecord(device, inputVirtArg);
		break;

	case VIRTQC_cudaEventSynchronize:
		status = qcu_cudaEventSynchronize(device, inputVirtArg);
		break;

	case VIRTQC_cudaEventElapsedTime:
		status = qcu_cudaEventElapsedTime(device, inputVirtArg);
		break;

	case VIRTQC_cudaEventDestroy:
		status = qcu_cudaEventDestroy(device, inputVirtArg);
		break;

		// Error Handling (runtime API)
	case VIRTQC_cudaGetLastError:
		status = qcu_cudaGetLastError(device, inputVirtArg);
		break;

		//Zero-copy
	case VIRTQC_cudaHostRegister:
		status = qcu_cudaHostRegister(device, inputVirtArg);
		break;

	case VIRTQC_cudaHostGetDevicePointer:
		status = qcu_cudaHostGetDevicePointer(device, inputVirtArg);
		break;

	case VIRTQC_cudaHostUnregister:
		status = qcu_cudaHostUnregister(device, inputVirtArg);
		break;

	case VIRTQC_cudaSetDeviceFlags:
		status = qcu_cudaSetDeviceFlags(device, inputVirtArg);
		break;

	case VIRTQC_cudaFreeHost:
		status = qcu_cudaFreeHost(device, inputVirtArg);
		if (NT_SUCCESS(status))
			inputVirtArg->cmd = 0;
		break;

		//stream
	case VIRTQC_cudaStreamCreate:
		status = qcu_cudaStreamCreate(device, inputVirtArg);
		break;

	case VIRTQC_cudaStreamDestroy:
		status = qcu_cudaStreamDestroy(device, inputVirtArg);
		break;
	default:
		// The specified I/O control code is unrecognized by this driver.
		//
		status = STATUS_INVALID_DEVICE_REQUEST;
		//TraceEvents(TRACE_LEVEL_ERROR, DBG_IOCTLS, "ERROR: unrecognized IOCTL %x\n", IoControlCode);
		break;
	}

	/*TraceEvents(TRACE_LEVEL_VERBOSE, DBG_IOCTLS, "Completing Request %p with status %X",
		Request, status);*/
	if (ioBuffer)
		RtlCopyMemory(ioBuffer, inputVirtArg, sizeof(VirtioQCArg));

	ExFreePoolWithTag(inputVirtArg, QCUDA_MEMORY_TAG);
	//irp->IoStatus.Status = status; irp->IoStatus.Information = OutputBufferLength;
	
	WdfRequestCompleteWithInformation(Request, status, OutputBufferLength);
	// IoCompleteRequest(irp, IO_NO_INCREMENT);
}
