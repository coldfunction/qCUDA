# 1 "multiboot.S"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "multiboot.S"
# 21 "multiboot.S"
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
# 22 "multiboot.S" 2
# 31 "multiboot.S"
.code16; .text; .global _start; _start:; .short 0xaa55; .byte (_end - _start) / 512; lret; .org 0x18; .short 0; .short _pnph; _pnph: .ascii "$PnP"; .byte 0x01; .byte ( _pnph_len / 16 ); .short 0x0000; .byte 0x00; .byte 0x00; .long 0x00000000; .short _manufacturer; .short _product; .long 0x00000000; .short 0x0000; .short 0x0000; .short _bev; .short 0x0000; .short 0x0000; .equ _pnph_len, . - _pnph; _bev:; movw %cs, %ax; movw %ax, %ds;

run_multiboot:

 cli
 cld

 mov %cs, %eax
 shl $0x4, %eax




 mov %ss, %ecx
 shl $0x4, %ecx
 mov %esp, %ebx
 add %ebx, %ecx
 sub $0x20, %ecx
 sub $0x30, %esp
 shr $0x4, %ecx
 mov %cx, %gs


 mov (prot_jump), %ebx
 add %eax, %ebx
 movl %ebx, %gs:0
 mov $8, %bx
 movw %bx, %gs:0 + 4


 movw (gdt_desc), %bx
 movw %bx, %gs:6
 movl (gdt_desc+2), %ebx
 add %eax, %ebx
 movl %ebx, %gs:6 + 2

 xor %eax, %eax
 mov %eax, %es


 read_fw 0x0a; mov %eax, %edi; read_fw 0x0b; mov %eax, %ecx; mov $0x12, %ax; mov $0x510, %edx; outw %ax, (%dx); mov $0x511, %dx; cld; .dc.b 0xf3,0x6c


 read_fw 0x0a
 shr $4, %eax
 mov %ax, %fs




 int $0x12
 cwtl
 movl %eax, %fs:4


 mov %fs:48, %eax
 shr $4, %eax
 mov %ax, %es


 xor %ebx, %ebx

 xor %edi, %edi

mmap_loop:

 add $4, %di

 movl $20, %ecx

 movl $0x0000e820, %eax

 movl $0x534d4150, %edx
 int $0x15

mmap_check_entry:

 jb mmap_done

mmap_store_entry:




 .dc.b 0x26,0x67,0x66,0x89,0x4f,0xfc


 add %ecx, %edi
 movw %di, %fs:0x2c


 test %ebx, %ebx
 jnz mmap_loop

mmap_done:


 xor %di, %di
 mov $0x100000, %edx
upper_mem_entry:
 cmp %fs:0x2c, %di
 je upper_mem_done
 add $4, %di


 cmpl $1, %es:16(%di)
 jne upper_mem_next


 movl %es:4(%di), %eax
 test %eax, %eax
 jnz upper_mem_next


 movl %es:(%di), %eax
 cmp %eax, %edx
 jb upper_mem_next
 addl %es:8(%di), %eax
 cmp %eax, %edx
 jae upper_mem_next


 mov %eax, %edx
 xor %di, %di
 jmp upper_mem_entry

upper_mem_next:
 addl %es:-4(%di), %edi
 jmp upper_mem_entry

upper_mem_done:
 sub $0x100000, %edx
 shr $10, %edx
 mov %edx, %fs:0x8

real_to_prot:

lgdt:
 data32 lgdt %gs:6


 movl $1, %eax
 movl %eax, %cr0


ljmp:
 data32 ljmp *%gs:0

prot_mode:
.code32


 movl $0x10, %eax
 movl %eax, %ss
 movl %eax, %ds
 movl %eax, %es
 movl %eax, %fs
 movl %eax, %gs


 read_fw 0x07; mov %eax, %edi; read_fw 0x08; mov %eax, %ecx; mov $0x11, %ax; mov $0x510, %edx; outw %ax, (%dx); mov $0x511, %dx; cld; .dc.b 0xf3,0x6c


 read_fw 0x10
 mov %eax, %ecx


 read_fw 0x0a
 movl %eax, %ebx


 movl $0x2badb002, %eax
ljmp2:
 jmp *%ecx


.align 4, 0
prot_jump: .long prot_mode
  .short 8

.align 4, 0
gdt:

.byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00


.byte 0xff, 0xff, 0x00, 0x00, 0x00, 0x9a, 0xcf, 0x00


.byte 0xff, 0xff, 0x00, 0x00, 0x00, 0x92, 0xcf, 0x00


.byte 0xff, 0xff, 0x00, 0x00, 0x00, 0x9e, 0x00, 0x00


.byte 0xff, 0xff, 0x00, 0x00, 0x00, 0x92, 0x00, 0x00

gdt_desc:
.short (5 * 8) - 1
.long gdt

_manufacturer:; .asciz "QEMU"; _product:; .asciz "multiboot loader"; .byte 0; .align 512, 0; _end:
