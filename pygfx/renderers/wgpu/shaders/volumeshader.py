import wgpu  # only for flags/enums

from ....objects import Volume
from ....materials import VolumeSliceMaterial, VolumeRayMaterial
from ....resources import Texture

from .. import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    to_texture_format,
    GfxSampler,
    GfxTextureView,
    load_wgsl,
)

vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


class BaseVolumeShader(BaseShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        geometry = wobject.geometry

        # Check grid
        if geometry.grid is None:
            raise ValueError("Volume.geometry must have a grid (texture).")
        elif not isinstance(geometry.grid, Texture):
            raise TypeError("Volume.geometry.grid must be a Texture")
        elif geometry.grid.dim != 3:
            raise TypeError("Volume.geometry.grid must a 3D texture (view)")

        # Set image format
        self["climcorrection"] = ""
        fmt = to_texture_format(geometry.grid.format)
        if "norm" in fmt or "float" in fmt:
            self["img_format"] = "f32"
            if "unorm" in fmt:
                self["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                self["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            self["img_format"] = "u32"
        else:
            self["img_format"] = "i32"

        # Set gamma
        self["gamma"] = material.gamma

        # Channels
        self["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

        # Colorspace
        self["colorspace"] = geometry.grid.colorspace
        if material.map is not None:
            self["colorspace"] = material.map.texture.colorspace

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        tex_view = GfxTextureView(geometry.grid)
        sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(Binding("s_img", "sampler/filtering", sampler, "FRAGMENT"))
        bindings.append(Binding("t_img", "texture/auto", tex_view, vertex_and_fragment))

        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }


@register_wgpu_render_function(Volume, VolumeSliceMaterial)
class VolumeSliceShader(BaseVolumeShader):
    type = "render"

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {
            "indices": (12, 1),
        }

    def get_code(self):
        return load_wgsl("volume_slice.wgsl")


@register_wgpu_render_function(Volume, VolumeRayMaterial)
class VolumeRayShader(BaseVolumeShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        render_mode = wobject.material.render_mode

        # Fall back to MIP, because we've written our examples to use the plain VolumeRayMaterial for quite a while.
        # Deprecate / remove this after a few releases (now is june 2025)
        render_mode = render_mode or "mip"

        if not render_mode:
            raise RuntimeError(
                f"Invalid value for {wobject.material.__class__.__name__}.render_mode: {render_mode!r}. Use an appropriate volume material, e.g. VolumeMipMaterial or VolumeIsoMaterial."
            )
        self["mode"] = render_mode
        return super().get_bindings(wobject, shared)

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,  # the back planes are the ref
        }

    def get_render_info(self, wobject, shared):
        return {
            "indices": (36, 1),
        }

    def get_code(self):
        return load_wgsl("volume_ray.wgsl")
