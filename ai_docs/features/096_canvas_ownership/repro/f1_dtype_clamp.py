import os
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE","4.6"); os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE","460")
import moderngl, numpy as np
from shaderbox.core import Canvas

gl = moderngl.create_standalone_context()
# One shader writing 3.0 -- above the f1 clamp ceiling.
vs="#version 330\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}"
fs="#version 330\nout vec4 c;void main(){c=vec4(3.0,0,0,1);}"
prog=gl.program(vertex_shader=vs,fragment_shader=fs)
vao=gl.vertex_array(prog,[(gl.buffer(np.array([-1,-1,3,-1,-1,3],dtype='f4')),'2f','p')])
for dt in ("f1","f2"):
    c = Canvas(gl=gl, size=(8,8), dtype=dt)
    c.fbo.use(); gl.clear(); vao.render()
    raw = np.frombuffer(c.texture.read(), dtype=('u1' if dt=='f1' else 'f2'))
    print(f"dest dtype={dt}: max R = {raw.max()}")
    c.release()
