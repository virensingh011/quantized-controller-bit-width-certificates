/* host test harness: reads N int8 vectors (CTRL_NIN each) from stdin, writes N int32 accumulators to stdout */
#include <stdio.h>
#include <stdint.h>
#include "ctrl_params.h"
int32_t ctrl_infer(const int8_t *x);
int main(void)
{
    int8_t v[CTRL_NIN];
    while (fread(v, 1, CTRL_NIN, stdin) == CTRL_NIN) { int32_t y = ctrl_infer(v); fwrite(&y, 4, 1, stdout); }
    return 0;
}
