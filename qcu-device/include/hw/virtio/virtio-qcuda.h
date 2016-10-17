
#ifndef _QEMU_VIRTIO_HM_H
#define _QEMU_VIRTIO_HM_H

#include "qemu/queue.h"
#include "hw/virtio/virtio.h"
#include "hw/pci/pci.h"

#define TYPE_VIRTIO_QC "virtio-qcuda-device"
#define VIRTIO_QC(obj)                                        \
        OBJECT_CHECK(VirtIOQC, (obj), TYPE_VIRTIO_QC)

//#define VIRTIO_ID_QCUDA 69

typedef struct VirtIOQCConf VirtIOQCConf;
typedef struct VirtIOQC VirtIOQC;

struct VirtIOQCConf
{
	uint64_t mem_size;
};

struct VirtIOQC
{
    VirtIODevice parent_obj;
	VirtIOQCConf conf;
	VirtQueue *vq;
};

#endif
