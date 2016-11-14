##qCUDA
qCUDA is based on the virtio framework to provide the para-virtualized driver as “front-end”, 
and the device module as “back-end” for performing the interaction with API remoting and memory management. 
In our test environment, qCUDA can achieve above 95% of the bandwidth efficiency for most results by comparing with the native. In addition, by comparing with prior work, qCUDA has more flexibility and interposition that it can execute CUDA-compatible programs in the Linux and Windows VMs, respectively, on QEMU-KVM hypervisor for GPGPU virtualization.

##System Components

The framework of qCUDA has three components, including qCUlibrary, qCUdriver and qCUdevice; the functions of 
these three components are defined as follows:

* qCUlibrary (`qcu-library`) – The interposer library in VM (guest OS) provided CUDA runtime access, interface of memory allocation, qCUDA command (qCUcmd), and passing the qCUcmd to the qCUdriver.

* qCUdriver (`qcu-driver`) – The front-end driver was responsible for the memory management, data movement, analyzing the qCUcmd from the qCUlibrary, and passing the qCUcmd by the control channel which is connected to the qCUdevice.

* qCUdevice (qcu-device) – The virtual device as the back-end was responsible for receiving/sending the qCUcmd through the control channel; it depended on receiving the qCUcmd to active related operations in the host, including to register GPU binary, convert guest physical addresses (GPA) into host virtual addresses (HVA), and handle the CUDA runtime/driver APIs for accessing the GPU.

##Installation
###Prerequisites
* CUDA 7.5
* Ubuntu 14.04.3 LTS (GNU/Linux 3.19.0-25-generic x86_64)
* Ubuntu 14.04 image (guest OS)
* Windows 8 image (guest OS)

###How to install
* qcu-device was modified from QEMU 2.4.0, under this forder add the "--enable-cuda option" in the configure: 
./configure --enable-cuda and then follow the installation steps of QEMU.

* Download qcu-driver and qcu-library in guest OS.
* Enter qcu-driver and execute the commands:
    * make all
    * make i
* Enter qcu-library and execute the commands:
    * make all
    * make install

## Contributors
* Yu-Shiang Lin
* Luis Herrera
* Jia-Chi Chen