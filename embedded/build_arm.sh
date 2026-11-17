#!/bin/sh
# Build the integer-only controller for a Cortex-M target and print the real code/data sizes.
# Needs the free "GNU Arm Embedded Toolchain" (arm-none-eabi-gcc).  Usage: ./build_arm.sh [cortex-m0plus|cortex-m3|cortex-m4]
# 1) generate the parameter header for a trained network, e.g. the bicycle seed 0 at 4 bits:
#       python gen_header.py bicycle 0 4          (writes ctrl_params.h here)
CPU=${1:-cortex-m0plus}
arm-none-eabi-gcc -mcpu=$CPU -mthumb -Os -ffunction-sections -fdata-sections -Wall -I. -c controller.c -o controller_$CPU.o
arm-none-eabi-size -A controller_$CPU.o
echo "text+rodata = flash; the activation buffer is 128 bytes of stack"
