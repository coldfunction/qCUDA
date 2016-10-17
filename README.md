# qCUDA

In start qemu vm's command, add "device virtio-hm-pci".


qcu-device:
	QEMU Hypervisior with virtio device. Using in host.

qcu-driver:
	virtio device driver. Using in guest.

qcu-library:
	cuda API. Using in guest.
	Note: before using this library, you should add "--enable-cuda" when compiler hypervisior. At Host side, it is using nvidia official driver.

test:
	sample code for write, read and cuda.
	
