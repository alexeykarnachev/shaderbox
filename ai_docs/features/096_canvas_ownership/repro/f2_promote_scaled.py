import os
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE","4.6"); os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE","460")
import moderngl
from shaderbox.document import Document
from shaderbox.core import Pass
from shaderbox.pass_graph import PassGraph, PassEntry, TargetConfig, PassSource

gl = moderngl.create_standalone_context()
PLAIN="#version 460 core\nin vec2 vs_uv;\nout vec4 fs_color;\nvoid main(){fs_color=vec4(vs_uv,0,1);}\n"
READER="#version 460 core\nin vec2 vs_uv;\nuniform sampler2D u_h;\nout vec4 fs_color;\nvoid main(){fs_color=texture(u_h,vs_uv);}\n"
doc=Document(gl=gl, canvas_size=(256,256))
for p in list(doc.passes.values()): p.release()
doc.passes={}
for n,s in (("helper",PLAIN),("main",READER)):
    p=Pass(gl=gl,canvas_size=(256,256),target=TargetConfig()); p.release_program(s); p.compile(); doc.passes[n]=p
doc.passes["main"].uniform_values["u_h"]=PassSource("helper")
# helper is a SCALED non-output pass
doc.graph=PassGraph(output="main", passes={"helper":PassEntry(target=TargetConfig(scale=0.5)),"main":PassEntry()})
doc.begin_frame(1); doc.render()
print(f"helper as scaled non-output: {doc.passes['helper'].canvas.texture.size}   (expect 128x128)")

# Promote helper to output -- the user clicks its tile. Through the Document VERB, which is
# what every production caller takes: the resize lives there, so assigning `graph.with_output`
# by hand still strands the pass and is not the path a click follows.
doc.set_output_pass("helper")
for f in (2,3,4):
    doc.begin_frame(f); doc.render()
print(f"helper AFTER promotion:      {doc.passes['helper'].canvas.texture.size}   (document is 256x256)")
print(f"what the viewer/export reads: {doc.render_pass.canvas.texture.size}")
print("BUG:", doc.render_pass.canvas.texture.size != doc.canvas_size)
