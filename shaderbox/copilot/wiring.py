from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import CopilotCapabilities


def build_capabilities(backend: CopilotBackend) -> CopilotCapabilities:
    # 1:1 bind of the backend's methods into the GL-free capabilities dataclass the agent/tools see.
    b = backend
    return CopilotCapabilities(
        node_tree=b.node_tree,
        lib_catalog=b.lib_catalog,
        template_catalog=b.template_catalog,
        read_shaders=b.read_shaders,
        grep=b.grep,
        read_lib=b.read_lib,
        read_working_set=b.read_working_set,
        batch_begin=b.batch_begin,
        apply_shader_edit=b.apply_shader_edit,
        apply_line_edit=b.apply_line_edit,
        set_uniform=b.set_uniform,
        create_node=b.create_node,
        delete_node=b.delete_node,
        switch_node=b.switch_node,
        render_image=b.render_image,
        render_video=b.render_video,
        publish_telegram=b.publish_telegram,
        publish_youtube=b.publish_youtube,
        has_current_node=b.has_current_node,
        telegram_connected=b.telegram_connected,
        youtube_connected=b.youtube_connected,
        telegram_has_default_pack=b.telegram_has_default_pack,
        set_telegram_token=b.set_telegram_token,
        telegram_connect=b.telegram_connect,
        telegram_token_set=b.telegram_token_set,
        list_telegram_packs=b.list_telegram_packs,
        select_telegram_pack=b.select_telegram_pack,
        create_telegram_pack=b.create_telegram_pack,
        delete_telegram_pack=b.delete_telegram_pack,
    )
