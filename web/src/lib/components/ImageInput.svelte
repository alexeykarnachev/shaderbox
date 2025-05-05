<script lang="ts">
    import { app_state } from "$lib/app_state.svelte";

    let prevent_default = (e: Event) => e.preventDefault();

    function load_base_image(file: File) {
        const reader = new FileReader();
        reader.onload = () => {
            app_state.base_image = (reader.result as string).split(",")[1];
        };
        reader.readAsDataURL(file);
    }

    function area_ondrop(e: DragEvent) {
        e.preventDefault();
        if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
            load_base_image(e.dataTransfer.files[0]);
        }
    }

    function input_onchange(e: Event) {
        const input = e.target as HTMLInputElement;
        if (input.files && input.files.length > 0) {
            load_base_image(input.files[0]);
        }
    }
</script>

<div id="image_input_container">
    <label
        for="file_input"
        ondragover={prevent_default}
        ondrop={area_ondrop}
        role="region"
        id="drop_area"
    >
        {#if app_state.base_image}
            <img src="data:image/png;base64,{app_state.base_image}" alt="Base" />
            <p>Base image</p>
        {:else}
            <p>Upload base image</p>
        {/if}
    </label>
    <input type="file" onchange={input_onchange} id="file_input" accept="image/*" />
</div>

<style>
    #image_input_container {
        display: flex;
        width: 300px;
        height: 400px;
    }

    #drop_area {
        display: flex;
        flex: 1;
        flex-direction: column;
        justify-content: flex-end;
        cursor: pointer;
        background: #202020;
        padding: 20px;
        overflow: hidden;
        color: #fff;
        text-align: center;
    }

    #drop_area img {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    #file_input {
        display: none;
    }
</style>
