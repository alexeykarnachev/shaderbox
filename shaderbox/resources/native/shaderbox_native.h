/*
 * shaderbox_native.h -- the contract between a ShaderBox script and a native library.
 *
 * Header-only. In exactly one source file of your library:
 *
 *     #include "shaderbox_native.h"
 *     SB_LIBRARY()
 *
 * Then export any number of functions of one shape:
 *
 *     SB_EXPORT int32_t step(sb_call *c) {
 *         sb_arg *vel = sb_buf(c, "vel");
 *         if (!vel || vel->dtype != SB_F32) return sb_fail(c, "vel must be an f4 buffer");
 *         double dt = sb_num(c, "dt", 1.0 / 60.0);
 *         ...
 *         sb_ret_num(c, "residual", 1e-4);
 *         return 0;
 *     }
 *
 * and call them from script.py with `native("mylib").call("step", vel=..., dt=...)`.
 *
 * Rules:
 *   - Every pointer in an sb_call (names, strings, buffer data) is valid for the duration of
 *     the call only. A buffer's CONTENTS persist across calls; its address does not across a
 *     reload, so keep no pointer between calls.
 *   - Your library's globals do not survive a rebuild: ShaderBox restarts the process that
 *     hosts it. State that must outlive a rebuild belongs in buffers.
 *   - Return 0 for success. Anything else is a failure; set its message with sb_fail.
 *   - A result string passed to sb_ret_str must stay valid until your export returns.
 *   - The optional export `int32_t sb_load(sb_call *c)` runs after every load, with
 *     nargs = 0 and args = NULL.
 *
 * Another language implements the same structs and exports `sb_abi`, `sb_layout_name` and
 * `sb_layout_value` by hand; ShaderBox compares that table, entry by entry, against its own
 * and refuses the library on the first difference.
 */
#ifndef SHADERBOX_NATIVE_H
#define SHADERBOX_NATIVE_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SB_ABI 1

#ifdef __cplusplus
#define SB_EXTERN extern "C"
#else
#define SB_EXTERN
#endif

#if defined(_WIN32)
#define SB_EXPORT SB_EXTERN __declspec(dllexport)
#else
#define SB_EXPORT SB_EXTERN __attribute__((visibility("default")))
#endif

enum { SB_NUMBER = 1, SB_INTEGER = 2, SB_STRING = 3, SB_BUFFER = 4 };           /* sb_arg.kind  */
enum { SB_F32 = 1, SB_F64 = 2, SB_I32 = 3, SB_U32 = 4, SB_I64 = 5, SB_U8 = 6 }; /* sb_arg.dtype */

typedef struct {
    const char *name;     /* UTF-8, NUL-terminated                                        */
    int32_t     kind;     /* SB_NUMBER .. SB_BUFFER                                        */
    int32_t     dtype;    /* SB_BUFFER only, else 0                                        */
    int32_t     ndim;     /* SB_BUFFER only: 1..4, else 0                                  */
    int32_t     reserved;
    int64_t     shape[4]; /* SB_BUFFER only: element counts, row-major; unused dims 0      */
    void       *data;     /* SB_BUFFER: C-contiguous elements; SB_STRING: UTF-8, NUL-terminated */
    double      number;   /* SB_NUMBER; for SB_INTEGER the value widened to double         */
    int64_t     integer;  /* SB_INTEGER, else 0                                            */
} sb_arg;

typedef struct {
    int32_t abi;          /* SB_ABI                                                        */
    int32_t nargs;
    sb_arg *args;
    char   *error;        /* sb_fail writes here, NUL-terminated                           */
    int32_t error_cap;    /* bytes, including the NUL; a longer message is truncated       */
    int32_t nresults;     /* written by sb_ret_*                                           */
    sb_arg *results;      /* filled by sb_ret_*: name, kind and number/integer/data        */
    int32_t results_cap;
    int32_t reserved;
} sb_call;

static inline sb_arg *sb_find(sb_call *c, const char *name) {
    for (int32_t i = 0; i < c->nargs; i++)
        if (strcmp(c->args[i].name, name) == 0) return &c->args[i];
    return NULL;
}

/* The buffer argument `name`, or NULL when it is absent or not a buffer. */
static inline sb_arg *sb_buf(sb_call *c, const char *name) {
    sb_arg *a = sb_find(c, name);
    return (a && a->kind == SB_BUFFER) ? a : NULL;
}

/* A number or an integer argument as a double; `fallback` when absent or of another kind. */
static inline double sb_num(sb_call *c, const char *name, double fallback) {
    sb_arg *a = sb_find(c, name);
    return (a && (a->kind == SB_NUMBER || a->kind == SB_INTEGER)) ? a->number : fallback;
}

static inline int64_t sb_int(sb_call *c, const char *name, int64_t fallback) {
    sb_arg *a = sb_find(c, name);
    return (a && a->kind == SB_INTEGER) ? a->integer : fallback;
}

static inline const char *sb_str(sb_call *c, const char *name, const char *fallback) {
    sb_arg *a = sb_find(c, name);
    return (a && a->kind == SB_STRING) ? (const char *)a->data : fallback;
}

/* Formats the failure message and returns 1, so `return sb_fail(c, ...)` reads naturally. */
static inline int32_t sb_fail(sb_call *c, const char *fmt, ...) {
    if (c->error && c->error_cap > 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(c->error, (size_t)c->error_cap, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static inline sb_arg *sb__result(sb_call *c, const char *name, int32_t kind) {
    if (!c->results || c->nresults >= c->results_cap) return NULL;
    sb_arg *r = &c->results[c->nresults++];
    memset(r, 0, sizeof *r);
    r->name = name;
    r->kind = kind;
    return r;
}

/* Named results, returned to the script as a dict. Each returns 0, or 1 when the call's
 * result slots are full. */
static inline int32_t sb_ret_num(sb_call *c, const char *name, double value) {
    sb_arg *r = sb__result(c, name, SB_NUMBER);
    if (!r) return 1;
    r->number = value;
    return 0;
}

static inline int32_t sb_ret_int(sb_call *c, const char *name, int64_t value) {
    sb_arg *r = sb__result(c, name, SB_INTEGER);
    if (!r) return 1;
    r->integer = value;
    r->number = (double)value;
    return 0;
}

static inline int32_t sb_ret_str(sb_call *c, const char *name, const char *value) {
    sb_arg *r = sb__result(c, name, SB_STRING);
    if (!r) return 1;
    r->data = (void *)value;
    return 0;
}

#define SB__ENTRY_SIZE(t) {"sizeof." #t, (int64_t)sizeof(t)}
#define SB__ENTRY_FIELD(t, f) {#t "." #f, (int64_t)offsetof(t, f)}
#define SB__ENTRY_ENUM(e) {#e, (int64_t)(e)}

/* Exports sb_abi and the layout table ShaderBox verifies at load. Write it once. */
#define SB_LIBRARY()                                                                       \
    static const struct { const char *name; int64_t value; } sb__layout[] = {              \
        SB__ENTRY_SIZE(sb_arg), SB__ENTRY_SIZE(sb_call),                                   \
        SB__ENTRY_FIELD(sb_arg, name), SB__ENTRY_FIELD(sb_arg, kind),                      \
        SB__ENTRY_FIELD(sb_arg, dtype), SB__ENTRY_FIELD(sb_arg, ndim),                     \
        SB__ENTRY_FIELD(sb_arg, reserved), SB__ENTRY_FIELD(sb_arg, shape),                 \
        SB__ENTRY_FIELD(sb_arg, data), SB__ENTRY_FIELD(sb_arg, number),                    \
        SB__ENTRY_FIELD(sb_arg, integer),                                                  \
        SB__ENTRY_FIELD(sb_call, abi), SB__ENTRY_FIELD(sb_call, nargs),                    \
        SB__ENTRY_FIELD(sb_call, args), SB__ENTRY_FIELD(sb_call, error),                   \
        SB__ENTRY_FIELD(sb_call, error_cap), SB__ENTRY_FIELD(sb_call, nresults),           \
        SB__ENTRY_FIELD(sb_call, results), SB__ENTRY_FIELD(sb_call, results_cap),          \
        SB__ENTRY_FIELD(sb_call, reserved),                                                \
        SB__ENTRY_ENUM(SB_NUMBER), SB__ENTRY_ENUM(SB_INTEGER),                             \
        SB__ENTRY_ENUM(SB_STRING), SB__ENTRY_ENUM(SB_BUFFER),                              \
        SB__ENTRY_ENUM(SB_F32), SB__ENTRY_ENUM(SB_F64), SB__ENTRY_ENUM(SB_I32),            \
        SB__ENTRY_ENUM(SB_U32), SB__ENTRY_ENUM(SB_I64), SB__ENTRY_ENUM(SB_U8),             \
    };                                                                                     \
    SB_EXPORT int32_t sb_abi(void) { return SB_ABI; }                                      \
    SB_EXPORT const char *sb_layout_name(int32_t i) {                                      \
        return (i >= 0 && (size_t)i < sizeof sb__layout / sizeof sb__layout[0])            \
                   ? sb__layout[i].name : NULL;                                            \
    }                                                                                      \
    SB_EXPORT int64_t sb_layout_value(int32_t i) {                                         \
        return (i >= 0 && (size_t)i < sizeof sb__layout / sizeof sb__layout[0])            \
                   ? sb__layout[i].value : 0;                                              \
    }

#endif
