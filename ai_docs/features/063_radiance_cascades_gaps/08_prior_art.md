# Audit: prior art in the maintainer's own repos

Source: a sweep of ~115 repos under the GitHub account `alexeykarnachev`, ~25 graphics repos
triaged, 12 deep-read. These repos are old and unreviewed — working reference, not gospel.
Defects are flagged as such rather than presented as patterns.

## The strongest precedent: freska's node graph

`freska` (C++/raylib) is a **texture-processing node graph** — structurally the closest thing
to what a ShaderBox multipass feature would be.

`src/graph.hpp` makes TEXTURE a first-class pin type:

```cpp
enum class PinType { INT, FLOAT, COLOR, TEXTURE, };
enum class PinKind { INPUT, OUTPUT, MANUAL, };
```

**`PinKind::MANUAL` is the key idea**: a third kind alongside INPUT/OUTPUT — a parameter with
min/max edited in the UI, which is *not* a graph edge. That maps 1:1 onto ShaderBox's
auto-generated uniform controls: a node's uniforms are MANUAL pins, its texture inputs are
INPUT pins, and they need not be the same mechanism.

`src/graph.cpp` — one node = one shader + one lazily-sized RenderTexture:

```cpp
class FrameProcessingContext : public NodeContext {
    Shader shader;
public:
    RenderTexture render_texture;

    FrameProcessingContext(std::string fs_file_name) {
        shader = load_shader("screen_rect.vert", fs_file_name);
        render_texture.id = 0;                  // sentinel: not yet allocated
    }

    void draw(std::vector<Pin> &pins) {
        Texture frame = pins[0]._texture;
        if (render_texture.id == 0 && IsTextureReady(frame)) {
            render_texture = LoadRenderTexture(frame.width, frame.height);  // size FROM INPUT
        }
        if (render_texture.id == 0 || !IsTextureReady(frame)) return;       // no input -> no-op
        BeginTextureMode(render_texture);
        BeginShaderMode(shader);
        set_shader_values(pins);   // every non-OUTPUT pin -> uniform, BY PIN NAME
        DrawRectangle(0, 0, 1, 1, BLANK);
        EndShaderMode();
        EndTextureMode();
    }
    void update(std::shared_ptr<Node> node) override {
        draw(node->pins);
        node->pins.back()._texture = render_texture.texture;   // publish output
    }
};
```

Note **the target size is derived from the input texture**, and a node with no ready input is
a silent no-op rather than an error. Uniforms bind by **pin name -> `GetShaderLocation`** — the
same introspection-driven binding ShaderBox already does natively.

A node type is then declared as just a shader filename plus a pin list — no per-node code.

**Two defects NOT to copy:**

1. **Evaluation order is broken, and he knew it.** `Graph::update()`:
   ```cpp
   // TODO: this is incorrect! Nodes must be sorted topologically
   for (auto [name, node] : this->nodes) { node->context->update(node); }
   for (auto &[_, link] : this->links) { /* copy start_pin value -> end_pin */ }
   ```
   Updating in `unordered_map` order and propagating links afterwards means a chain of N nodes
   **lags by up to N-1 frames**, nondeterministically. Any ShaderBox version must topologically
   sort. (The deleted ShaderBox DAG got ordering right via pull-recursion but lacked
   memoization — between them, the correct shape is *topological sort with memoization*.)
2. Inputs addressed by hard index (`pins[0]._texture`), with his own TODO to key them by name.

**freska is the evolution of `webcam_filters`**, which implemented the identical four effects
as ONE uber-shader with uniform structs and zero render targets. He explicitly moved
uber-shader -> per-node RenderTexture graph when he wanted arbitrary composition. That is a
direct precedent for the same move here.

## The established buffer shape (C engines)

One consistent shape across `coxel` and `crossover`: a `*Buffer` struct = `fbo` + `rbo` +
named textures + `width`/`height`, one `<name>_create(...)` function, **always unbound back to
0 at the end**. No destroy, no resize, no bind method.

`coxel`'s `GBuffer` is the MRT one (5 attachments: `GL_RGB32F` world pos and normals,
`GL_RGBA32F` diffuse, `GL_R32F` specular, `GL_R8UI` entity-id for picking) — so **he has
written float render targets before**, just never in a moderngl project.

Passes are always **hardcoded comment-banner blocks** in one function:
`bind -> viewport -> clear -> draw`, with a fullscreen `glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)`
resolve (no VBO — positions generated in `screen_rect.vert`). Never a list of pass objects,
never a graph, except freska.

## Best ideas worth lifting

- **`simg`'s pass helpers** (Rust/glow) — the tidiest render-target API in the corpus:
  ```rust
  fn bind_framebuffer(gl, fbo: Option<NativeFramebuffer>, viewport, color: Option<Color>, depth: bool)
  fn blit_framebuffer(gl, src, dst: Option<...>, src_rect, dst_rect)
  ```
  `Option<Framebuffer>` where `None` means the screen; bind + viewport + clear in one call.
  And postfx as `Option<&Program>` passed per frame — **the pass exists only if you hand it a
  shader, otherwise it degrades to a blit.** The cheapest "optional extra pass" shape he has.
- **`soft_tissues`' pass jobs** — the only data-driven pass list: a `ShadowPassJob` POD
  (`entity`, `camera`, `shadow_map`) produced by a prepare step and consumed by a dumb loop,
  with the *same* scene-draw functions serving both passes, differentiated by a
  `RenderState.is_shadow_map_pass` flag rather than duplicated code. Plus a **render-target
  pool** with explicit exhaustion logging, and a **`needs_update` dirty flag** so a STATIC
  light bakes its shadow map once and never again — directly applicable to RC's
  cached-distance-field-invalidated-on-draw.

## Clean negatives

- **Ping-pong / double buffering: zero hits across every repo.** He has never written one.
  There is no shape here to reuse.
- **Bloom / downsample / mip chains: none.** No multi-resolution anything.
- **JFA / jump flood: none. Radiance cascades: none.**
- **SDF baked into a texture: never** — SDFs exist only as per-pixel analytic fragment code.
- **moderngl multi-pass: none.** `py2glsl` is his only other moderngl project and is
  single-pass (one optional offscreen FBO purely for headless export). **ShaderBox would be
  the first.**
- `shader_sandbox`, the direct ShaderBox ancestor (C++/raylib/imgui, hot-reload,
  `example_NNN_*.frag.glsl`), has **zero** render targets — the same architecture ShaderBox
  has today.

2D lighting exists twice, both geometry-rasterized visibility rather than distance-field:
`crossover`'s additive light-mask pass (`GL_ONE, GL_ONE` blend accumulating per-light vision
circles) and `no_dungeon_no_dragons`'s CPU shadow-volume triangles rasterized white into a
512x512 map.

**Flagged as a defect, not a pattern:** `lift`'s "HDR pipeline" renders to an `RGBA32F` fbo and
resolves with a shader whose entire body is `frag_color = texture(tex, vs_uv);` — no tone
mapping, no exposure. The HDR buffer buys nothing as written. It is an unfinished stub.

## The recurring weakness to fix up front

Across coxel, crossover, and freska: **no resize path** (buffers created once at fixed size),
**no destroy**, and per-input uniform binding copy-pasted with his own "factor this out" TODOs
left unactioned in all three. He has never solved resize or named-input binding.

Whatever ShaderBox builds should solve both up front — ShaderBox already has `Canvas.set_size`
(release + reallocate) and GL-introspection uniform binding, so it starts ahead of every one of
these repos on exactly the two axes they all failed.
