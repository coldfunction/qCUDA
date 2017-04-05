##qCuda Windows Client
The imposter cudart library intercepts the CUDA API calls and then uses virtio to forward the commands
to the driver. The driver then decides what parameters to package based on the IOCTL code. The driver then uses
virtio to push the command onto a queue which the hypervisor listens to and performs the necessary operations before
returning the results to the client driver using virtio.

###How to install
* After installing Windows 7 OS or later version you must enable large-page support.
* The cudart library path needs to be added to the Path environmental variable.
* This driver is not signed; therefore, Windows needs to be booted with driver signature verification disabled.
* The driver can then be installed using the device manager.

###How to compile
In the following command, replace the compute architecure with those your GPGPU allows
* nvcc -m64 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86\_amd64" {cuda\_source}.cu -cudart=shared -gencode arch=compute\_x,code=sm\_x -o binary.exe

