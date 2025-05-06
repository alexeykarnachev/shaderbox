<script lang="ts">
    let { file = $bindable(null) }: { file: File | null } = $props();
    let img_src: string | null = $state(null);

    let prevent_default = (e: Event) => e.preventDefault();

    function area_ondrop(e: DragEvent) {
        e.preventDefault();
        if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
            file = e.dataTransfer.files[0];
        }
    }

    function input_onchange(e: Event) {
        const input = e.target as HTMLInputElement;
        if (input.files && input.files.length > 0) {
            file = input.files[0];
        }
    }

    $effect(() => {
        if (img_src !== null) {
            URL.revokeObjectURL(img_src);
        }

        if (file !== null) {
            img_src = URL.createObjectURL(file);
        }
    });
</script>

<div id="image_input_container">
    <label
        for="file_input"
        ondragover={prevent_default}
        ondrop={area_ondrop}
        role="region"
        id="drop_area"
    >
        {#if img_src !== null}
            <img src={img_src} alt="" />
        {:else}
            <p>Drag and drop an image here or click to select</p>
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
        justify-content: center;
        align-items: center;
        cursor: pointer;
        background: #202020;
        padding: 20px;
        overflow: hidden;
        color: #fff;
        text-align: center;
    }

    #drop_area img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }

    #file_input {
        display: none;
    }
</style>
