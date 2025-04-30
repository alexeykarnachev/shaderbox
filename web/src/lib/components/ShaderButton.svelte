<script lang="ts">
    export let text = "";
    export let on_click = () => {};
</script>

<button on:click={on_click} class="shader_button">
    {text}
</button>

<style>
    .shader_button {
        cursor: pointer;
        border: none;
        background-color: #333;
        padding: 6px 12px;
        color: white;
    }

    .shader_button:hover {
        background-color: #444;
    }

    .shader_button:active {
        background-color: #222;
    }

    .shader_button:focus {
        outline: 1px solid #888;
    }

    .shader_button:focus:not(:focus-visible) {
        outline: none;
    }
</style>
