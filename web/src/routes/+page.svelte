<script lang="ts">
    import ImageInput from "$lib/components/ImageInput.svelte";
    import ShaderButton from "$lib/components/ShaderButton.svelte";

    let base_image_file: File | null = null;

    async function shader_button_onclick(shader_name: string) {
        if (base_image_file === null) return;

        let body = JSON.stringify({ name: shader_name, image: image });
        await fetch("/api/apply_shader", {
            method: "POST",
            body: body
        });
    }
</script>

<div class="flex flex-row gap-1">
    <ImageInput bind:file={base_image_file} />

    <div class="flex flex-col gap-1">
        <ShaderButton
            text="Normals Map"
            onclick={() => shader_button_onclick("normals_map")}
        />
        <ShaderButton
            text="Green Scan"
            onclick={() => shader_button_onclick("green_scan")}
        />
    </div>
</div>
