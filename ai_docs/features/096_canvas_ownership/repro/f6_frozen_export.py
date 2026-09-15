import os
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE","4.6"); os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE","460")
import moderngl, numpy as np, imageio.v3 as iio
from pathlib import Path
from shaderbox.document import Document
from shaderbox.core import Pass
from shaderbox.pass_graph import PassGraph, PassEntry, TargetConfig, PassSource
from shaderbox.media import MediaDetails, FileDetails, ResolutionDetails

gl = moderngl.create_standalone_context()
SRC=("#version 460 core\nin vec2 vs_uv;\nuniform sampler2D u_prev;\nout vec4 fs_color;\n"
     "void main(){ fs_color = texture(u_prev, vs_uv) + vec4(0.05,0,0,1); }\n")
out = Path("/tmp/claude-1000/-home-akarnachev-src-shaderbox/3116e4dc-736d-4180-bf68-8b06724b3e32/scratchpad/fb.mp4")
for iters in (1,2):
    doc=Document(gl=gl, canvas_size=(64,64))
    for p in list(doc.passes.values()): p.release()
    doc.passes={}
    p=Pass(gl=gl,canvas_size=(64,64),target=TargetConfig()); p.release_program(SRC); p.compile()
    doc.passes["acc"]=p; doc.passes["acc"].uniform_values["u_prev"]=PassSource("acc")
    doc.graph=PassGraph(output="acc", passes={"acc":PassEntry(iterations=iters)})
    det = MediaDetails(is_video=True, file_details=FileDetails(path=str(out), size=0),
                       resolution_details=ResolutionDetails(width=64,height=64), fps=10, duration=0.6)
    doc.render_media(det)
    frames = iio.imread(out)
    reds = [int(f[:,:,0].max()) for f in frames]
    print(f"iterations={iters}: exported reds per frame -> {reds}")
