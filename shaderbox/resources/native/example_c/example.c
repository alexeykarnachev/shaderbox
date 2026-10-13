/*
 * A minimal ShaderBox native library in C. Build, from this directory:
 *
 *     cc -shared -fPIC -O2 -I.. -o example.so example.c        (Linux)
 *     cl /LD /O2 /I.. example.c /Fe:example.dll                (Windows, MSVC)
 *
 * put the library next to a document's script.py, and in the script:
 *
 *     self.lib = native("example")
 *     self.values = self.lib.buffer(1024, "f4")
 *     total = self.lib.call("scale", values=self.values, factor=2.0)["sum"]
 */
#include "shaderbox_native.h"

SB_LIBRARY()

SB_EXPORT int32_t scale(sb_call *c) {
    sb_arg *values = sb_buf(c, "values");
    if (!values || values->dtype != SB_F32) return sb_fail(c, "values must be an f4 buffer");
    float factor = (float)sb_num(c, "factor", 1.0);
    int64_t n = 1;
    for (int32_t d = 0; d < values->ndim; d++) n *= values->shape[d];
    float *v = (float *)values->data;
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        v[i] *= factor;
        sum += v[i];
    }
    sb_ret_num(c, "sum", sum);
    return 0;
}
