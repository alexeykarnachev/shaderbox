"""FINAL: the whole RC pipeline inside the REAL ShaderBox engine, as a script.py,
with the node's own shader.frag.glsl doing the final gather. Zero engine changes.

Also measures: does the engine's per-frame overhead (script tick + Node.render)
leave room at 60fps, and does the node's unconditional clear() break anything?
"""
import os, time
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE","4.6"); os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE","460")
import tempfile, pathlib, moderngl, numpy as np
from PIL import Image
from shaderbox.core import Node
from shaderbox.scripting import EngineContext, ScriptEngine

ctx = moderngl.create_standalone_context()
W=H=512

# The NODE's shader: the final gather + tonemap. NOTE: hand-rolled SDFs, NOT SB_* --
# resolve_usage runs only inside Node.compile(), so a script's own gl.program() gets NO lib splicing.
NODE_SRC = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_cascade0;
uniform float u_exposure;
out vec4 fs_color;
void main(){
    vec3 c = texture(u_cascade0, vs_uv).rgb * u_exposure;
    vec2 p = vs_uv;
    vec2 q = abs(p-vec2(0.50,0.50)) - vec2(0.020,0.30);
    float occ = min(length(max(q,vec2(0.0)))+min(max(q.x,q.y),0.0),
                    length(p-vec2(0.22,0.26))-0.07);
    if(occ < 0.0) c = vec3(0.02);
    c = c/(1.0+c);
    fs_color = vec4(pow(c, vec3(1.0/2.2)), 1.0);
}
"""

node = Node(gl=ctx)
node.release_program(NODE_SRC)
node.compile()
print("node compile errors:", node.compile_unit.errors if node.compile_unit else "n/a")
node.canvas.set_size((W,H))

tmp = pathlib.Path(tempfile.mkdtemp()); sd = tmp/"scripts"; sd.mkdir()

SCRIPT = r'''
import sys, moderngl, numpy as np
QUAD=np.array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1],dtype="f4")
VS="""#version 460 core
in vec2 in_pos; out vec2 vs_uv;
void main(){vs_uv=in_pos*0.5+0.5;gl_Position=vec4(in_pos,0,1);}"""
SCENE="""
float sdc(vec2 p,float r){return length(p)-r;}
float sdb(vec2 p,vec2 b){vec2 q=abs(p)-b;return length(max(q,vec2(0.0)))+min(max(q.x,q.y),0.0);}
float occl(vec2 p){float d=sdb(p-vec2(0.50,0.50),vec2(0.020,0.30));
  return min(d,sdc(p-vec2(0.22,0.26),0.07));}
float lightf(vec2 p,float t){
  float d=sdc(p-vec2(0.28,0.70+0.06*sin(t)),0.04);
  return min(d,sdb(p-vec2(0.80,0.80),vec2(0.06,0.014)));}
float scene(vec2 p,float t){return min(lightf(p,t),occl(p));}
vec3 emis(vec2 p,float t){
  if(sdc(p-vec2(0.28,0.70+0.06*sin(t)),0.04)<0.0) return vec3(4.0,3.3,2.0);
  if(sdb(p-vec2(0.80,0.80),vec2(0.06,0.014))<0.0) return vec3(0.5,1.2,4.0);
  return vec3(0.0);}
"""
FS_RC="""#version 460 core
in vec2 vs_uv; out vec4 fs_color;
uniform sampler2D u_up; uniform float u_c,u_count,u_t,u_res;
"""+SCENE+"""
const float TAU=6.28318530718;
vec4 march(vec2 o,vec2 d,float t0,float t1,float t){
  float s=t0;
  for(int i=0;i<96;i++){
    vec2 p=o+d*s;
    if(p.x<0.0||p.x>1.0||p.y<0.0||p.y>1.0) return vec4(0,0,0,1);
    float h=scene(p,t);
    if(h<0.0005) return vec4(emis(p,t),1.0);
    s+=max(h,0.0005);
    if(s>t1) return vec4(0.0);}
  return vec4(0.0);}
void main(){
  float sp=exp2(u_c), rays=4.0*sp*sp;
  vec2 co=floor(vs_uv*u_res);
  vec2 pr=floor(co/sp), sl=mod(co,sp);
  float si=sl.x+sl.y*sp;
  vec2 pUv=(pr+0.5)*sp/u_res;
  float BL=0.012;
  float t0=(u_c==0.0)?0.0:BL*(pow(4.0,u_c)-1.0)/3.0;
  float t1=BL*(pow(4.0,u_c+1.0)-1.0)/3.0;
  vec3 acc=vec3(0.0); float usp=sp*2.0;
  vec2 upf=pUv*u_res/usp-0.5;
  for(float k=0.0;k<4.0;k++){
    float idx=si*4.0+k;
    float a=TAU*(idx+0.5)/rays;
    vec4 hit=march(pUv,vec2(cos(a),sin(a)),t0,t1,u_t);
    vec3 r=hit.rgb;
    if(hit.a<0.5 && u_c<u_count-1.0){
      vec3 m=vec3(0.0);
      for(float j=0.0;j<4.0;j++){
        float ui=idx*4.0+j;
        float uS=mod(ui,usp*usp);
        vec2 uSlot=vec2(mod(uS,usp),floor(uS/usp));
        vec2 base=floor(upf), fr=fract(upf);
        vec2 lim=vec2(u_res/usp-1.0);
        vec2 c00=clamp(base,vec2(0.0),lim)*usp+uSlot+0.5;
        vec2 c10=clamp(base+vec2(1,0),vec2(0.0),lim)*usp+uSlot+0.5;
        vec2 c01=clamp(base+vec2(0,1),vec2(0.0),lim)*usp+uSlot+0.5;
        vec2 c11=clamp(base+vec2(1,1),vec2(0.0),lim)*usp+uSlot+0.5;
        m+=mix(mix(texture(u_up,c00/u_res).rgb,texture(u_up,c10/u_res).rgb,fr.x),
               mix(texture(u_up,c01/u_res).rgb,texture(u_up,c11/u_res).rgb,fr.x),fr.y);}
      r+=m*0.25;}
    acc+=r;}
  fs_color=vec4(acc*0.25,1.0);}
"""
RES=512; C=6

class Behavior(ScriptBehavior):
    def __init__(self):
        self.gl=None; self.ms=0.0

    def _node(self):
        f=sys._getframe(); d=0
        while f is not None and d<20:
            n=f.f_locals.get("node")
            if n is not None and hasattr(n,"uniform_values") and hasattr(n,"canvas"): return n
            f=f.f_back; d+=1
        return None

    def _setup(self):
        gl=moderngl.get_context(); self.gl=gl
        self.vbo=gl.buffer(QUAD.tobytes())
        self.prog=gl.program(vertex_shader=VS,fragment_shader=FS_RC)
        self.vao=gl.vertex_array(self.prog,[(self.vbo,"2f","in_pos")])
        self.tex=[];self.fbo=[]
        for i in range(2):
            t=gl.texture((RES,RES),4,dtype="f2")
            t.filter=(moderngl.LINEAR,moderngl.LINEAR); t.repeat_x=False;t.repeat_y=False
            self.tex.append(t); self.fbo.append(gl.framebuffer(color_attachments=[t]))

    def update(self, ctx):
        import time as _t
        if self.gl is None: self._setup()
        gl=self.gl; t0=_t.perf_counter()
        src=0
        for c in range(C-1,-1,-1):
            dst=1-src
            self.fbo[dst].use(); gl.clear()
            self.tex[src].use(0)
            self.prog["u_up"]=0; self.prog["u_c"]=float(c); self.prog["u_count"]=float(C)
            self.prog["u_t"]=ctx.t; self.prog["u_res"]=float(RES)
            self.vao.render(); src=dst
        gl.finish(); self.ms=(_t.perf_counter()-t0)*1000.0
        n=self._node()
        if n is not None:
            n.uniform_values["u_cascade0"]=self.tex[src]
        return {"u_exposure":1.0}
'''
(sd/"script.py").write_text(SCRIPT)
eng=ScriptEngine(); eng.reload("n1", sd, node)
print("script compile errors:", {k:v.message for k,v in eng.errors.items()})

inst=eng._nodes["n1"].behavior._instance
N=60; t0=time.perf_counter()
for i in range(N):
    eng.tick("n1", node, EngineContext(t=i/60.0, dt=1/60, frame=i))
    node.render(u_time=i/60.0)
ctx.finish(); t1=time.perf_counter()
print("tick errors:", {k:v.message for k,v in eng.errors.items()})
print(f"FULL engine loop (script multipass + Node.render): {(t1-t0)/N*1000:.2f} ms/frame -> {N/(t1-t0):.0f} fps @{W}x{H}")
print(f"  of which the cascade chain alone: {inst.ms:.2f} ms")
img=np.frombuffer(node.canvas.texture.read(),dtype=np.uint8).reshape(H,W,4)
Image.fromarray(img[::-1]).save("/tmp/claude-1000/-home-akarnachev-src-shaderbox/1813baee-fc8a-4631-84e8-8f89a2f19822/scratchpad/final.png")
print("node canvas mean:",img[...,:3].mean().round(1))
