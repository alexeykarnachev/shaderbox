<script lang="ts">
    import ImageInput from "$lib/components/ImageInput.svelte";
    import ShaderButton from "$lib/components/ShaderButton.svelte";

    let base_image_file: File | null = null;

    async function shader_button_onclick(name: string) {
        if (base_image_file === null) return;

        try {
            let form_data = new FormData();
            form_data.append("image", base_image_file);
            form_data.append("name", name);

            let response = await fetch("/api/apply_node", {
                method: "POST",
                body: form_data
            });
            console.log(response);

            if (!response.ok) {
                throw new Error();
            }

            let result = await response.json();
        } catch (err) {
            console.error("Failed to apply shader");
        }
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
