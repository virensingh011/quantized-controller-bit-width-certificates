/* Integer-only inference of the quantized MLP controller (3 hidden ReLU layers, scalar output).
   Weights: signed b-bit integers, packed LSB-first, stored output-channel-major (W^T).
   Activations: input int8 (|a|<=127), hidden uint8 (0..255); accumulators int32;
   requantization with a per-channel fixed-point multiplier (Jacob et al., CVPR 2018, zero points = 0).
   Returns the int32 accumulator of the output layer; the control is  u = acc * CTRL_OUT_SCALE  (then saturated). */
#include <stdint.h>
#include "ctrl_params.h"

typedef struct { uint8_t nin, nout; const uint8_t *w; const int32_t *b; const int32_t *m; const uint8_t *s; } layer_t;
static const layer_t LAY[4] = {
    { CTRL_NIN, 32, L0_W, L0_B, L0_M, L0_S },
    { 32,       32, L1_W, L1_B, L1_M, L1_S },
    { 32,       32, L2_W, L2_B, L2_M, L2_S },
    { 32,        1, L3_W, L3_B, 0,    0    } };

static inline int32_t wget(const uint8_t *p, uint32_t idx)
{
    uint32_t bit = idx * CTRL_BITS;
    uint32_t v = p[bit >> 3] | ((uint32_t)p[(bit >> 3) + 1] << 8);   /* arrays carry one padding byte */
    v = (v >> (bit & 7u)) & ((1u << CTRL_BITS) - 1u);
    return (int32_t)(v ^ (1u << (CTRL_BITS - 1))) - (int32_t)(1u << (CTRL_BITS - 1));   /* sign extension */
}

int32_t ctrl_infer(const int8_t *x)
{
    int16_t buf[2][32];                       /* 128 bytes of RAM for activations */
    int cur = 0;
    for (uint32_t i = 0; i < CTRL_NIN; i++) buf[0][i] = x[i];
    for (int l = 0; l < 4; l++) {
        const layer_t *L = &LAY[l];
        for (uint32_t j = 0; j < L->nout; j++) {
            int32_t acc = L->b[j];
            uint32_t base = j * L->nin;
            for (uint32_t i = 0; i < L->nin; i++) acc += (int32_t)buf[cur][i] * wget(L->w, base + i);
            if (l == 3) return acc;
            if (acc < 0) acc = 0;                                    /* ReLU */
            int sh = L->s[j];
            int64_t q = ((int64_t)acc * L->m[j] + ((int64_t)1 << (sh - 1))) >> sh;
            buf[cur ^ 1][j] = (int16_t)(q > 255 ? 255 : q);
        }
        cur ^= 1;
    }
    return 0;
}
