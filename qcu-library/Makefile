
CUDA_INC=/usr/local/cuda/include

LIB=libcudart

all: lib

libcudart_dl:
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -o libcudart.so $(LIB).o -lcuda -ldl

libcudart:
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -o libcudart.so $(LIB).o

lib: $(LIB)

install: 
	sudo cp libcudart.so /usr/local/lib/
	sudo ln -sf libcudart.so /usr/local/lib/libcudart.so.7.5
	sudo rm -f /etc/ld.so.conf.d/cuda-7.5.conf
	sudo ldconfig

clean:
	rm -f *.so *.o 

remove:
	sudo rm -f /usr/local/lib/libcudart.so /usr/local/lib/libcudart.so.7.5
	sudo su -c "echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda-7.5.conf"
	sudo ldconfig

#/usr/local/cuda/bin/nvcc --cudart=shared $(USR).c
