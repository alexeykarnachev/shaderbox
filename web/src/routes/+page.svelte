<script lang="ts">
    import { app_state } from "$lib/app_state.svelte";
    import ShaderButton from "$lib/components/ShaderButton.svelte";

    $effect(() => {
        console.log(app_state.current_page);
    });
</script>

<div class="flex flex-row">
    <div
        class="flex h-screen min-w-60 flex-col gap-1 border border-red-500 p-1 text-2xl"
    >
        <ShaderButton
            text="Depth Map"
            on_click={() => (app_state.current_page = "shader_depth_map")}
        />
        <ShaderButton
            text="Normals Map"
            on_click={() => (app_state.current_page = "shader_normals_map")}
        />
        <ShaderButton
            text="Green Scan"
            on_click={() => (app_state.current_page = "shader_green_scan")}
        />
    </div>
</div>
