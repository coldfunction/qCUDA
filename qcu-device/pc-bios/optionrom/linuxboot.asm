# 1 "linuxboot.S"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "linuxboot.S"
# 23 "linuxboot.S"
# 1 "optionrom.h" 1
# 23 "optionrom.h"
# 1 "../../include/hw/nvram/fw_cfg.h" 1
# 24 "optionrom.h" 2
# 38 "optionrom.h"
.macro read_fw VAR
 mov $\VAR, %ax
 mov $0x510, %dx
 outw %ax, (%dx)
 mov $0x511, %dx
 inb (%dx), %al
 shl $8, %eax
 inb (%dx), %al
 shl $8, %eax
 inb (%dx), %al
 shl $8, %eax
 inb (%dx), %al
 bswap %eax
.endm
# 24 "linuxboot.S" 2



.code16; .text; .global _start; _start:; .short 0xaa55; .byte (_end - _start) / 512; lret; .org 0x18; .short 0; .short _pnph; _pnph: .ascii "$PnP"; .byte 0x01; .byte ( _pnph_len / 16 ); .short 0x0000; .byte 0x00; .byte 0x00; .long 0x00000000; .short _manufacturer; .short _product; .long 0x00000000; .short 0x0000; .short 0x0000; .short _bev; .short 0x0000; .short 0x0000; .equ _pnph_len, . - _pnph; _bev:; movw %cs, %ax; movw %ax, %ds;

run_linuxboot:

 cli
 cld

 jmp copy_kernel
boot_kernel:

 read_fw 0x16

 mov %eax, %ebx
 shr $4, %ebx


 mov %bx, %ds
 mov %bx, %es
 mov %bx, %fs
 mov %bx, %gs
 mov %bx, %ss


 add $0x20, %bx
 mov %bx, %cx


 read_fw 0x13
 mov %eax, %ebx
 read_fw 0x16
 sub %eax, %ebx
 sub $16, %ebx
 mov %ebx, %esp


 pushw %cx
 xor %ax, %ax
 pushw %ax


 xor %eax, %eax
 xor %ebx, %ebx
 xor %ecx, %ecx
 xor %edx, %edx
 xor %edi, %edi
 xor %ebp, %ebp


 lret


copy_kernel:

 read_fw 0x16
 shr $4, %eax
 mov %eax, %es
 xor %edi, %edi
 read_fw 0x17; mov %eax, %ecx; mov $0x18, %ax; mov $0x510, %edx; outw %ax, (%dx); mov $0x511, %dx; cld; .dc.b 0x67,0xf3,0x6c

 cmpw $0x203, %es:0x206
 jae 1f
 movl $0x37ffffff, %es:0x22c
1:


 read_fw 0x0a
 mov %eax, %edi
 read_fw 0x0b
 add %edi, %eax
 xor %es:0x22c, %eax
 and $-4096, %eax
 jz load_kernel





 mov $0xe801, %ax
 xor %cx, %cx
 xor %dx, %dx
 int $0x15


 or %cx, %cx
 jnz 1f
 or %dx, %dx
 jnz 1f
 mov %ax, %cx
 mov %bx, %dx
1:

 or %dx, %dx
 jnz 2f
 addw $1024, %cx
 movzwl %cx, %edi
 shll $10, %edi
 jmp 3f

2:
 addw $16777216 >> 16, %dx
 movzwl %dx, %edi
 shll $16, %edi

3:
 read_fw 0x0b
 subl %eax, %edi
 andl $-4096, %edi
 movl %edi, %es:0x218

load_kernel:





 mov %esp, %ebp
 sub $16, %esp


 movw $((3 * 8) - 1), -16(%bp)
 mov %cs, %eax
 movzwl %ax, %eax
 shl $4, %eax
 addl $gdt, %eax
 movl %eax, -14(%bp)


 data32 lgdt -16(%bp)
 mov %ebp, %esp


 mov $1, %eax
 mov %eax, %cr0


 mov $0x10, %eax
 mov %eax, %es




 read_fw 0x0b; mov %eax, %ecx; mov $0x12, %ax; mov $0x510, %edx; outw %ax, (%dx); mov $0x511, %dx; cld; .dc.b 0x67,0xf3,0x6c
 read_fw 0x07; mov %eax, %edi; read_fw 0x08; mov %eax, %ecx; mov $0x11, %ax; mov $0x510, %edx; outw %ax, (%dx); mov $0x511, %dx; cld; .dc.b 0x67,0xf3,0x6c
 read_fw 0x13; mov %eax, %edi; read_fw 0x14; mov %eax, %ecx; mov $0x15, %ax; mov $0x510, %edx; outw %ax, (%dx); mov $0x511, %dx; cld; .dc.b 0x67,0xf3,0x6c


 mov $0, %eax
 mov %eax, %cr0


 mov %cs, %ax
 mov %ax, %es

 jmp boot_kernel



.align 4, 0
gdt:

.byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00


.byte 0xff, 0xff, 0x00, 0x00, 0x00, 0x9a, 0xcf, 0x00


.byte 0xff, 0xff, 0x00, 0x00, 0x00, 0x92, 0xcf, 0x00

_manufacturer:; .asciz "QEMU"; _product:; .asciz "Linux loader"; .byte 0; .align 512, 0; _end:
