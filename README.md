## qCUDA
qCUDA is based on the virtio framework to provide the para-virtualized driver as “front-end”, 
and the device module as “back-end” for performing the interaction with API remoting and memory management. 
In our test environment, qCUDA can achieve above 95% of the bandwidth efficiency for most results by comparing with the native. In addition, by comparing with prior work, qCUDA has more flexibility and interposition that it can execute CUDA-compatible programs in the Linux and Windows VMs, respectively, on QEMU-KVM hypervisor for GPGPU virtualization.

## System Components

The framework of qCUDA has three components, including qCUlibrary, qCUdriver and qCUdevice; the functions of 
these three components are defined as follows:

* qCUlibrary (`qcu-library`) – The interposer library in VM (guest OS) provided CUDA runtime access, interface of memory allocation, qCUDA command (qCUcmd), and passing the qCUcmd to the qCUdriver.

* qCUdriver (`qcu-driver`) – The front-end driver was responsible for the memory management, data movement, analyzing the qCUcmd from the qCUlibrary, and passing the qCUcmd by the control channel which is connected to the qCUdevice.

* qCUdevice (`qcu-device`) – The virtual device as the back-end was responsible for receiving/sending the qCUcmd through the control channel; it depended on receiving the qCUcmd to active related operations in the host, including to register GPU binary, convert guest physical addresses (GPA) into host virtual addresses (HVA), and handle the CUDA runtime/driver APIs for accessing the GPU.

## Installation

### Prerequisites

#### Host

* CUDA 7.5
* Ubuntu 14.04.3 LTS (GNU/Linux 3.19.0-25-generic x86_64)
* export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
* export CUDA_HOME=/usr/local/cuda
* Install required packages
  
	``` sh
	sudo apt-get install -y  pkg-config bridge-utils uml-utilities zlib1g-dev libglib2.0-dev autoconf
	automake libtool libsdl1.2-dev libsasl2-dev libcurl4-openssl-dev libsasl2-dev libaio-dev libvde-dev
	```

#### Guest

* Ubuntu 14.04 image (guest OS)
* Windows 8 image (guest OS)
 
### How to install

#### Host

* `qcu-device` was modified from QEMU 2.4.0, for further information please refer to [QEMU installation steps](https://en.wikibooks.org/wiki/QEMU/Installing_QEMU)

1. clone this repo.
2. cd qcu-device
3. ./configure --enable-cuda --target-list=x86_64-softmmu  && make -j16
4. sudo mkdir /dev/qcuvf
5. sudo chmod 777 /dev/qcuvf

#### Guest

1. clone this repo.
2. Enter `qcu-driver` and execute the commands:
    * make all
    * make i
3. Enter `qcu-library` and execute the commands:
    * make all
    * make install


## How to use qCUDA framework

In our current version, qCUDA has been implementing for 32 CUDA runtime APIs. These implemented CUDA runtime APIs on qCUDA are shown in the table as below:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;border-top-width:1px;border-bottom-width:1px;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;border-top-width:1px;border-bottom-width:1px;}
.tg .tg-yw4l{vertical-align:top}
.tg .tg-3we0{background-color:#ffffff;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">Classification</th>
    <th class="tg-yw4l">CUDA runtime API on qCUDA</th>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="5">Memory Management</td>
    <td class="tg-3we0">cudaMalloc</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemset</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemcpy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemcpyAsync</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaFree</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="6">Device Management</td>
    <td class="tg-3we0">cudaGetDevice</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaGetDeviceCount</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaSetDevice</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaGetDeviceProperties</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaDeviceSynchronize</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaDeviceReset</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="2">Version Management</td>
    <td class="tg-3we0">cudaDriverGetVersion</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaRuntimeGetVersion</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="3">Stream Management</td>
    <td class="tg-3we0">cudaStreamCreate</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaStreamDestroy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaStreamSynchronize</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="6">Event Management</td>
    <td class="tg-3we0">cudaEventCreate</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventCreateWithFlags</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventRecord</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventSynchronize</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventElapsedTime</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventDestroy</td>
  </tr>
  <tr>
    <td class="tg-yw4l">Error Handling</td>
    <td class="tg-3we0">cudaGetLastError</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="4">Zero-copy</td>
    <td class="tg-3we0">cudaHostRegister</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaHostGetDevicePointer</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaHostUnregister</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaSetDeviceFlags</td>
  </tr>
  <tr>
    <td class="tg-yw4l">Thread Management</td>
    <td class="tg-3we0">cudaThreadSynchronize</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="4">Module &amp; Execution Control</td>
    <td class="tg-3we0">cudaRegisterFatBinary</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaUnregisterFatBinary</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaRegisterFunction</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaLaunch</td>
  </tr>
</table>


According to our design, it is very easy to add the new CUDA runtime API via our framework. It only need to modify three parts in qCUDA, the main components we have talked in [previous section](https://github.com/coldfunction/qCUDA#system-components), qCUlibrary, qCUdriver and qCUdevice; these three components of qCUDA source code are located on "qcu-library/libcudart.c", "qcu-driver/qcuda_driver.c" and "qcu-device/hw/misc/virtio-qcuda.c". If a programmer wants to add a new CUDA API that user in guest OS can use the new function, she/he should follow the standards of qCUDA framework to modify these files. We gave an example to show how to add a CUDA runtime API, "cudaThreadSynchronize", modified related files on the qCUDA system, described as below:

### qcu-library/libcudart.c 
The qCUlibrary component of qCUDA system, providing the interface to wrap the CUDA runtime APIs. The CUDA application in guest can link the function that implemented in the "libcudart.c". It shows how to add the CUDA function "cudaThreadSynchronize" as below:


``` C
cudaError_t cudaThreadSynchronize () {
    VirtioQCArg arg ;
    memset(&arg , 0, sizeof (VirtioQCArg ));
    send_cmd_to_device ( VIRTQC_cudaThreadSynchronize, &arg );
    return ( cudaError_t ) arg .cmd;
}
```
The qCUDA command, qCUcmd, is represented by *VirtioQCArg*, which is the structure defined in "qcuda_common.h"; we can take this structure as the buffer to pass the variety of parameters that you want to interact with qCUdriver. The structure *VirtioQCArg* defined as below:

``` C
typedef struct VirtioQCArg VirtioQCArg;

struct VirtioQCArg {
    int32_t cmd;
    uint64_t rnd;
    uint64_t para;
    
    uint64_t pA;
    uint32_t pASize;
    
    uint64_t pB;
    uint32_t pBSize;
    
    uint32_t flag;    
};
 ```
 
In this function, "cudaThreadSynchronize", just only has the void parameter; thus we don’t need to pass any parameter to driver, but we could receive the returned value from driver. We use *send_cmd_to_device* there to send qCUcmd to qCUdriver; the first parameter of *send_cmd_to_device* is the identified name of cudaThreadSynchronize, should be add the prefix "*VIRTQC_*" of the head of function name; the second parameter is the *VirtioQCArg* structure we just declared and initialized it. After the send_cmd_to_device called, we can get the returned value from the specific member of *VirtioQCArg*.

### qcu-library/qcuda_driver.c
The qCUdriver component of qCUDA system, providing the driver interface of guest. Through the driver module, guest can pass the message to the virtual device. It shows how to modified the qCUdriver below:

``` C
// @_cmd: device command
// @_arg: argument of cuda function
// this function return cudaError_t .
static long qcu_misc_ioctl(struct file *filp, 
                            unsigned int _cmd, 
                            unsigned long _arg)
{
    VirtioQCArg *arg ; 
    int err ;
    
    arg = kmalloc_safe(sizeof(VirtioQCArg));
    copy_from_user_safe(arg , \
                    (void*) _arg, \
                    sizeof(VirtioQCArg ));
    arg−>cmd = _cmd
    
    switch( arg−>cmd )
    {
        ....
        case VIRTQC_cudaThreadSynchronize:
            qcu_cudaThreadSynchronize ( arg );
            break ; 
        ....
    } 

```

Add the new condition, "*VIRTQCcudaThreadSynchronize*", in the switch case
of the *qcu_misc_ioctl* function, then next line add "*qcucudaThreadSynchronize(arg)*" as the function call. Note the prefix "*VIRTQC_*" and "*qcu_*" must be add in the head of our function name. Next we implemented the "*qcucudaThreadSynchronize(arg)*" as below:

```C
void qcu_cudaThreadSynchronize ( VirtioQCArg *arg ) {
    qcu_misc_send_cmd ( arg ); 
}

```

In this case, it is very simple that we don’t need add extra information in driver, just call the function "*qcu_misc_send_cmd*" for passing the qCUcmd to the qCUdevice.


### qcu-device/hw/misc/virtio-qcuda.c

The qCUdevice component of qCUDA system, providing the interface to execute the actual CUDA runtime API and pass message to guest. Different with the other components, it defined in host and implemented in the part of QEMU source. It shows how to modify the qCUdevice below:

```C

static void virtio_qcuda_cmd_handle ( VirtIODevice *vdev ,
                                      VirtQueue *vq)
{
    VirtQueueElement elem;
    VirtioQCArg *arg;
    
    arg = malloc( sizeof(VirtioQCArg)); 
    while(virtqueue_pop(vq, &elem))
    {
        iov_to_buf(elem.out_sg , \ 
                    elem . out_num , \
                    0,\
                    arg, \
                    sizeof (VirtioQCArg ));
        
        ...
        
        case VIRTQC_cudaThreadSynchronize :
            qcu_cudaThreadSynchronize ( arg ); 
            break ;
    
        ...
    
    }
    
    ...
    
}
```


Add the new condition, "case VIRTQC_cudaThreadSynchronize", in the switch case of the *virtio_qcuda_cmd_handle* function, then next line add "*qcucudaThreadSynchronize(arg)*" as the function call. Note the prefix "*VIRTQC*" and "*qcu_*" must be add in the head of our function name. Next we implemented the "*qcu_cudaThreadSynchronize(arg)*" as below:

```C
static void qcu_cudaThreadSynchronize(VirtioQCArg *arg)
{
    cudaError_t err ;
    cudaError(err = cudaThreadSynchronize());
    
    arg−>cmd = err;
}
```

It is very simple that we just add the CUDA function we need here, the *cudaThreadSynchronize* could return the value, then we can pass the value from the specific entry defined of the *VirtioQCArg* structure.

### qcuda_common.h

The common arguments and macro defined here, use the enumeration to define as below, we must add the prefix "VIRTQC_" of the head of the function name as the identified name.

```C
enum
{
    // Module & Execution control (driver API)
    VIRTQC_cudaRegisterFatBinary = 200,
    VIRTQC_cudaUnregisterFatBinary,
    VIRTQC_cudaRegisterFunction,
    VIRTQC_cudaRegisterVar,
    VIRTQC_cudaLaunch,

    // Memory Management (runtime API)
    VIRTQC_cudaMalloc,
    VIRTQC_cudaMemcpy,
    VIRTQC_cudaMemcpyAsync,
    VIRTQC_cudaMemset,
    VIRTQC_cudaFree,

    // Device Management (runtime API)
    VIRTQC_cudaGetDevice,
    VIRTQC_cudaGetDeviceCount,
    VIRTQC_cudaGetDeviceProperties,
    VIRTQC_cudaSetDevice,
    VIRTQC_cudaDeviceSynchronize,
    VIRTQC_cudaDeviceReset,
    VIRTQC_cudaDeviceSetLimit,

    // Version Management (runtime API)
    VIRTQC_cudaDriverGetVersion,
    VIRTQC_cudaRuntimeGetVersion,

    // Event Management (runtime API)
    VIRTQC_cudaEventCreate,
    VIRTQC_cudaEventCreateWithFlags,
    VIRTQC_cudaEventRecord,
    VIRTQC_cudaEventSynchronize,
    VIRTQC_cudaEventElapsedTime,
    VIRTQC_cudaEventDestroy,

    // Error Handling (runtime API)
    VIRTQC_cudaGetLastError,

    //zero-cpy
    VIRTQC_cudaHostRegister,
    VIRTQC_cudaHostGetDevicePointer,
    VIRTQC_cudaHostUnregister,
    VIRTQC_cudaSetDeviceFlags,
    VIRTQC_cudaFreeHost,

    //stream
    VIRTQC_cudaStreamCreate,
    VIRTQC_cudaStreamDestroy,
    VIRTQC_cudaStreamSynchronize,

    // Thread Management
    VIRTQC_cudaThreadSynchronize,
};

```

According to the above sample, we know that through the qCUDA framwork, programmer doesn’t care about the details of the path of qCUcmd passing between guest and host; also the details of VirtIO are hidden from our high level of abstraction interface.





## Contributors
* Yu-Shiang Lin
* Luis Herrera
* Jia-Chi Chen
