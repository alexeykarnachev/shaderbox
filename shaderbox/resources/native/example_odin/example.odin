// A minimal ShaderBox native library in Odin. Build, from this directory:
//
//     odin build . -build-mode:shared -o:speed -out:example.so      (example.dll on Windows)
//
// put example.so next to a document's script.py, and in the script:
//
//     self.lib = native("example")
//     self.values = self.lib.buffer(1024, "f4")
//     total = self.lib.call("scale", values=self.values, factor=2.0)["sum"]
package example

import "base:runtime"
import sb "../shaderbox_native"

@(export)
scale :: proc "c" (c: ^sb.Call) -> i32 {
	context = runtime.default_context()
	values := sb.buf(c, "values")
	if values == nil || values.dtype != sb.F32 {
		return sb.fail(c, "values must be an f4 buffer")
	}
	factor := f32(sb.num(c, "factor", 1.0))
	sum: f64 = 0
	for &v in sb.elements(values, f32) {
		v *= factor
		sum += f64(v)
	}
	sb.ret_num(c, "sum", sum)
	return 0
}
