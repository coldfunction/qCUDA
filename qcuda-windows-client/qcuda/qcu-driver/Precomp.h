#pragma once
#include <stddef.h>
#include <stdarg.h>
#include <ntddk.h>
#include <wdf.h>

#include <ntstrsafe.h>
#include <wdmsec.h>
//#include <sddl.h>
//#include <aclapi.h>
#include <wmistr.h>
#include <wmilib.h>
#include <math.h>

#include "osdep.h"

#include "virtio_pci.h"
#include "virtio_config.h"
#include "virtio.h"

#ifdef _MSC_VER
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#else
#include <stdint.h>
#endif

#include "../qcuda/qcuda_common.h"
#include "ProtoTypes.h"
#include "Trace.h"