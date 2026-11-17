# Integer-only C controller and size measurement

`controller.c` is an integer-only inference routine for the quantized 3x32 MLP controller (b-bit packed weights,
int8/uint8 activations, int32 accumulators, fixed-point requantization after Jacob et al., CVPR 2018). 
* `export_c.py`: exports a trained, rounded network as `ctrl_params.h` and a bit-exact NumPy model (`fx_forward`).
* `verify_c.py`: (1) checks that the C code (host gcc) equals the NumPy model on 6,000 points at 3, 4, 5, 8 bits;
  (2) closed-loop sampling test of the integer-only controller for 15 seeds; (3) constant-data sizes. Results: `res_embedded.json`.
* `gen_header.py <plant> <seed> <bits>` writes `ctrl_params.h`; `build_arm.sh [cortex-m0plus|cortex-m3|cortex-m4]` compiles for a
  Cortex-M target with `arm-none-eabi-gcc` and prints real code and data sizes.

Not done here (no ARM toolchain or board was available): code size and timing on a real target. The unpack routine handles
b <= 8 bits. The input quantization (a_i = round(127 x_i / s_i)) and the final scaling u = acc * CTRL_OUT_SCALE (then saturation)
are done by the caller.
