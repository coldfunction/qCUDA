#include "Precomp.h"

#if defined(EVENT_TRACING)
#include "driver.tmh"
#endif

#ifdef ALLOC_PRAGMA
#pragma alloc_text( INIT, DriverEntry )
#pragma alloc_text( PAGE, QcudaEvtDriverContextCleanup)
#pragma alloc_text( PAGE, QcudaEvtDriverUnload)
#endif // ALLOC_PRAGMA

NTSTATUS
DriverEntry(
	IN OUT PDRIVER_OBJECT   DriverObject,
	IN PUNICODE_STRING      RegistryPath
	)
{
	WDF_DRIVER_CONFIG      config;
	NTSTATUS               status = STATUS_SUCCESS;
	WDFDRIVER              driver;
	WDF_OBJECT_ATTRIBUTES  attrib;
	// PWDFDEVICE_INIT        pInit = NULL;

	//WPP_INIT_TRACING(DriverObject, RegistryPath); 
	KdPrint(("Starting driver\n"));
	
	/*TraceEvents(TRACE_LEVEL_WARNING, DBG_HW_ACCESS, "Qcuda driver, built on %s %s\n",
		__DATE__, __TIME__);*/

	WDF_DRIVER_CONFIG_INIT(
		&config,
		QcudaDeviceAdd
		);
/*
	config.DriverInitFlags |= WdfDriverInitNonPnpDriver;

	config.EvtDriverUnload = QcudaEvtDriverUnload;*/

	WDF_OBJECT_ATTRIBUTES_INIT(&attrib);
	attrib.EvtCleanupCallback = QcudaEvtDriverContextCleanup;

	// WDF_DRIVER_CONFIG_INIT(&config, QcudaDeviceAdd);

	status = WdfDriverCreate(
		DriverObject,
		RegistryPath,
		&attrib,
		&config,
		&driver);
	if (!NT_SUCCESS(status))
	{
		/*TraceEvents(TRACE_LEVEL_ERROR, DBG_PNP, "WdfDriverCreate failed with status 0x%08x\n", status);
		WPP_CLEANUP(DriverObject);*/
		return status;
	}

	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_PNP, "<-- %s\n", __FUNCTION__);

	//pInit = WdfControlDeviceInitAllocate(
	//	driver,
	//	&SDDL_DEVOBJ_SYS_ALL_ADM_RWX_WORLD_RW_RES_R
	//	);

	//if (pInit == NULL) {
	//	status = STATUS_INSUFFICIENT_RESOURCES;
	//	return status;
	//}

	//status = QcudaDeviceAdd(driver, pInit);

	return status;
}

VOID
QcudaEvtDriverContextCleanup(
	IN WDFOBJECT Driver
	)
{
	UNREFERENCED_PARAMETER(Driver);
	PAGED_CODE();

	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_PNP, "--> %s\n", __FUNCTION__);

	//WPP_CLEANUP(WdfDriverWdmGetDriverObject((WDFDRIVER)Driver));

	//TraceEvents(TRACE_LEVEL_INFORMATION, DBG_PNP, "<-- %s\n", __FUNCTION__);
}

VOID
QcudaEvtDriverUnload(
	IN WDFDRIVER Driver
	)
{
	UNREFERENCED_PARAMETER(Driver);

	PAGED_CODE();

	//TraceEvents(TRACE_LEVEL_VERBOSE, DBG_POWER, "Entered NonPnpDriverUnload\n");

	return;
}