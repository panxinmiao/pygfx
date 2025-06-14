import wgpu  # only for flags/enums

from ....objects import Text
from ....materials import TextMaterial
from ....utils.text._shaper import REF_GLYPH_SIZE

from .. import (
    register_wgpu_render_function,
    load_wgsl,
    BaseShader,
    Binding,
    GfxSampler,
    GfxTextureView,
)


@register_wgpu_render_function(Text, TextMaterial)
class TextShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        self["is_screen_space"] = wobject.screen_space
        self["is_multi_text"] = wobject.is_multi
        self["aa"] = material.aa
        self["REF_GLYPH_SIZE"] = REF_GLYPH_SIZE

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        sbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_positions", sbuffer, geometry.positions, "VERTEX"),
        ]

        tex = shared.glyph_atlas_texture
        sampler = GfxSampler("linear", "clamp")
        tex_view = GfxTextureView(tex)
        bindings.append(Binding("s_atlas", "sampler/filtering", sampler, "FRAGMENT"))
        bindings.append(Binding("t_atlas", "texture/auto", tex_view, "FRAGMENT"))

        # Let the shader generate code for our bindings
        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        bindings1 = {}
        bindings1[0] = Binding(
            "s_glyph_info", sbuffer, shared.glyph_atlas_info_buffer, "VERTEX"
        )
        bindings1[1] = Binding("s_glyph_data", sbuffer, geometry.glyph_data, "VERTEX")

        return {
            0: bindings,
            1: bindings1,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        n = wobject.geometry.glyph_data.draw_range[1] * 6
        return {
            "indices": (n, 1),
        }

    def get_code(self):
        return load_wgsl("text.wgsl")
