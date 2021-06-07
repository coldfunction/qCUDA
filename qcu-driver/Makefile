# Makefile 2.6

# KernelKeyPath := ./kernel_key.pem # kernel key path

MOD	:= qcuda

obj-m := ${MOD}.o
${MOD}-objs := ${MOD}_driver.o

#CFLAGS_MODULE+=-I/usr/local/cuda/include
KERDIR:=/lib/modules/$(shell uname -r)/build
PWD=$(shell pwd)

all: c

c compile:
	make -C $(KERDIR) M=$(PWD) modules
	# $(KERDIR)/scripts/sign-file sha256 $(KernelKeyPath) $(KernelKeyPath) $(PWD)/$(MOD).ko

clean:
	make -C $(KERDIR) M=$(PWD) clean
	rm -f modules.order *.o *.mod.c *.mod.o *.ko *.o.ur-safe

i insmod:
	@if [ -n "`lsmod | grep $(MOD)`" ]; then \
	sudo rmmod $(MOD);  \
	echo "sudo rmmod $(MOD)"; \
	fi
	sudo dmesg -C
	sudo insmod $(MOD).ko
	-sudo chmod 666 /dev/qcuda

r rmmod:
	@if [ -n "`lsmod | grep $(MOD)`" ]; then \
	sudo rmmod $(MOD);  \
	echo "sudo rmmod $(MOD)"; \
	fi

s show_dmesg:
	@echo "============================================================="
	-@dmesg
	@echo "============================================================="

