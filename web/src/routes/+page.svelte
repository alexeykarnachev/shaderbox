<script lang="ts">
    import ImageInput from "$lib/components/ImageInput.svelte";
    import ShaderButton from "$lib/components/ShaderButton.svelte";

    let base_image_file: File | null = null;

    function shader_button_onclick(shader_name: string) {}
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
