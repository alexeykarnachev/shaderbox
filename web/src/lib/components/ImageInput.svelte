<script lang="ts">
    let selected_file = $state<File | null>(null);

    let prevent_default = (e: Event) => e.preventDefault();

    function area_ondrop(e: DragEvent) {
        e.preventDefault();
        if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
            selected_file = e.dataTransfer.files[0];
        }
    }

    function input_onchange(e: Event) {
        const input = e.target as HTMLInputElement;
        if (input.files && input.files.length > 0) {
            selected_file = input.files[0];
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
        Select or drop a file
    </label>
    <input type="file" onchange={input_onchange} id="file_input" />
</div>

<style>
    #image_input_container {
        width: 300px;
    }

    #drop_area {
        display: block;
        cursor: pointer;
        border: 2px dashed #ccc;
        background: #0000ff;
        padding: 20px;
        color: #fff;
        text-align: center;
    }

    #file_input {
        display: none;
    }
</style>
