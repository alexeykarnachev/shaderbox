import os
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE","4.6"); os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE","460")
import tempfile
import moderngl, numpy as np, imageio.v3 as iio
from pathlib import Path
from shaderbox.document import Document
from shaderbox.core import Pass
from shaderbox.pass_graph import PassGraph, PassEntry, TargetConfig, PassSource
from shaderbox.media import MediaDetails, FileDetails, ResolutionDetails

gl = moderngl.create_standalone_context()
SRC=("#version 460 core\nin vec2 vs_uv;\nuniform sampler2D u_prev;\nout vec4 fs_color;\n"
     "void main(){ fs_color = texture(u_prev, vs_uv) + vec4(0.05,0,0,1); }\n")
# A throwaway file in the system temp dir: this script must run from any session, so it
# cannot reference a scratchpad path that belonged to the one that wrote it.
out = Path(tempfile.gettempdir()) / "shaderbox_096_f6.mp4"
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
