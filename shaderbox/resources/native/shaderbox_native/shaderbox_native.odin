// The ShaderBox native-library contract for Odin: the same ABI as shaderbox_native.h.
//
// Import this package from your library; importing it exports `sb_abi` and the layout table
// ShaderBox verifies at load. Then export procedures of one shape:
//
//     @(export)
//     step :: proc "c" (c: ^sb.Call) -> i32 {
//         context = runtime.default_context()   // required before calling ordinary Odin code
//         ...
//         return 0
//     }
//
// Every pointer in a Call is valid for the duration of the call only, and your package's
// globals do not survive a rebuild. See shaderbox_native.h for the full rules.
package shaderbox_native

import "core:fmt"

ABI :: 1

NUMBER :: 1
INTEGER :: 2
STRING :: 3
BUFFER :: 4

F32 :: 1
F64 :: 2
I32 :: 3
U32 :: 4
I64 :: 5
U8 :: 6

Arg :: struct {
	name:     cstring,
	kind:     i32,
	dtype:    i32,
	ndim:     i32,
	reserved: i32,
	shape:    [4]i64,
	data:     rawptr,
	number:   f64,
	integer:  i64,
}

Call :: struct {
	abi:         i32,
	nargs:       i32,
	args:        [^]Arg,
	error:       [^]u8,
	error_cap:   i32,
	nresults:    i32,
	results:     [^]Arg,
	results_cap: i32,
	reserved:    i32,
}

@(private)
Entry :: struct {
	name:  cstring,
	value: i64,
}

@(private)
LAYOUT := [?]Entry {
	{"sizeof.sb_arg", size_of(Arg)},
	{"sizeof.sb_call", size_of(Call)},
	{"sb_arg.name", i64(offset_of(Arg, name))},
	{"sb_arg.kind", i64(offset_of(Arg, kind))},
	{"sb_arg.dtype", i64(offset_of(Arg, dtype))},
	{"sb_arg.ndim", i64(offset_of(Arg, ndim))},
	{"sb_arg.reserved", i64(offset_of(Arg, reserved))},
	{"sb_arg.shape", i64(offset_of(Arg, shape))},
	{"sb_arg.data", i64(offset_of(Arg, data))},
	{"sb_arg.number", i64(offset_of(Arg, number))},
	{"sb_arg.integer", i64(offset_of(Arg, integer))},
	{"sb_call.abi", i64(offset_of(Call, abi))},
	{"sb_call.nargs", i64(offset_of(Call, nargs))},
	{"sb_call.args", i64(offset_of(Call, args))},
	{"sb_call.error", i64(offset_of(Call, error))},
	{"sb_call.error_cap", i64(offset_of(Call, error_cap))},
	{"sb_call.nresults", i64(offset_of(Call, nresults))},
	{"sb_call.results", i64(offset_of(Call, results))},
	{"sb_call.results_cap", i64(offset_of(Call, results_cap))},
	{"sb_call.reserved", i64(offset_of(Call, reserved))},
	{"SB_NUMBER", NUMBER},
	{"SB_INTEGER", INTEGER},
	{"SB_STRING", STRING},
	{"SB_BUFFER", BUFFER},
	{"SB_F32", F32},
	{"SB_F64", F64},
	{"SB_I32", I32},
	{"SB_U32", U32},
	{"SB_I64", I64},
	{"SB_U8", U8},
}

@(export)
sb_abi :: proc "c" () -> i32 {
	return ABI
}

@(export)
sb_layout_name :: proc "c" (i: i32) -> cstring {
	if i < 0 || int(i) >= len(LAYOUT) {
		return nil
	}
	return LAYOUT[i].name
}

@(export)
sb_layout_value :: proc "c" (i: i32) -> i64 {
	if i < 0 || int(i) >= len(LAYOUT) {
		return 0
	}
	return LAYOUT[i].value
}

find :: proc(c: ^Call, name: string) -> ^Arg {
	for i in 0 ..< int(c.nargs) {
		if string(c.args[i].name) == name {
			return &c.args[i]
		}
	}
	return nil
}

// The buffer argument `name`, or nil when it is absent or not a buffer.
buf :: proc(c: ^Call, name: string) -> ^Arg {
	a := find(c, name)
	return a if a != nil && a.kind == BUFFER else nil
}

// A number or an integer argument as f64; `fallback` when absent or of another kind.
num :: proc(c: ^Call, name: string, fallback: f64) -> f64 {
	a := find(c, name)
	return a.number if a != nil && (a.kind == NUMBER || a.kind == INTEGER) else fallback
}

integer :: proc(c: ^Call, name: string, fallback: i64) -> i64 {
	a := find(c, name)
	return a.integer if a != nil && a.kind == INTEGER else fallback
}

str :: proc(c: ^Call, name: string, fallback: string) -> string {
	a := find(c, name)
	return string(cstring(a.data)) if a != nil && a.kind == STRING else fallback
}

// The element count of a buffer argument.
count :: proc(a: ^Arg) -> int {
	n := 1
	for d in 0 ..< int(a.ndim) {
		n *= int(a.shape[d])
	}
	return n
}

// A buffer argument's elements as a slice, valid for the duration of the call.
elements :: proc(a: ^Arg, $T: typeid) -> []T {
	return ([^]T)(a.data)[:count(a)]
}

// Formats the failure message and returns 1, so `return sb.fail(c, ...)` reads naturally.
fail :: proc(c: ^Call, format: string, args: ..any) -> i32 {
	if c.error != nil && c.error_cap > 0 {
		written := fmt.bprintf(c.error[:c.error_cap - 1], format, ..args)
		c.error[len(written)] = 0
	}
	return 1
}

@(private)
result :: proc(c: ^Call, name: cstring, kind: i32) -> ^Arg {
	if c.results == nil || c.nresults >= c.results_cap {
		return nil
	}
	r := &c.results[c.nresults]
	c.nresults += 1
	r^ = {}
	r.name = name
	r.kind = kind
	return r
}

// Named results, returned to the script as a dict. Each returns 0, or 1 when the result
// slots are full. A string result must stay valid until the export returns.
ret_num :: proc(c: ^Call, name: cstring, value: f64) -> i32 {
	r := result(c, name, NUMBER)
	if r == nil {
		return 1
	}
	r.number = value
	return 0
}

ret_int :: proc(c: ^Call, name: cstring, value: i64) -> i32 {
	r := result(c, name, INTEGER)
	if r == nil {
		return 1
	}
	r.integer = value
	r.number = f64(value)
	return 0
}

ret_str :: proc(c: ^Call, name: cstring, value: cstring) -> i32 {
	r := result(c, name, STRING)
	if r == nil {
		return 1
	}
	r.data = rawptr(value)
	return 0
}
