
CUDA_INC=/usr/local/cuda/include

LIB=libcudart

all: lib

libcudart_dl:
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -o libcudart.so $(LIB).o -lcuda -ldl

libcudart:
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -Wl,--version-script=shared.version -o libcudart.so.9.0 $(LIB).o

lib: $(LIB)

install: 
	sudo cp libcudart.so.9.0 /usr/local/lib/libcudart.so.9.0
	sudo ln -sf libcudart.so.9.0 /usr/local/lib/libcudart.so
	sudo ldconfig

clean:
	rm -f *.so *.o *.so.9.0

remove:
	sudo rm -f /usr/local/lib/libcudart.so /usr/local/lib/libcudart.so.9.0
	sudo su -c "echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda-9.0.conf"
	sudo ldconfig

#/usr/local/cuda/bin/nvcc --cudart=shared $(USR).c
