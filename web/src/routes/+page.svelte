<script lang="ts">
    import { app_state } from "$lib/app_state.svelte";
    import ImageInput from "$lib/components/ImageInput.svelte";
    import ShaderButton from "$lib/components/ShaderButton.svelte";

    $effect(() => {
        console.log(app_state.current_page);
    });
</script>

<div class="flex flex-row">
    <ImageInput />
</div>
