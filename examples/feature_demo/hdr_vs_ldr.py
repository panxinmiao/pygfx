"""
HDR vs LDR Rendering
====================

This example shows a comparison between HDR and LDR rendering.
"""

################################################################################
# .. note::
#
#   To run this example, you need a model from the source repo's example
#   folder. If you are running this example from a local copy of the code (dev
#   install) no further actions are needed. Otherwise, you may have to replace
#   the path below to point to the location of the model.

import os
import numpy as np
from pathlib import Path

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "data"


################################################################################
# Once the path is set correctly, you can use the model as follows:

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx

# Init
canvas = RenderCanvas(size=(800, 480), title="HDR vs LDR")

hdr_renderer = gfx.WgpuRenderer(canvas, hdr=True)
ldr_renderer = gfx.WgpuRenderer(canvas, hdr=False)

scene = gfx.Scene()

# Create bloom effect pass using the new API
bloom_pass = gfx.renderers.wgpu.PhysicalBasedBloomPass(
    bloom_strength=0.05,
    max_mip_levels=6,
    filter_radius=0.003,
    use_karis_average=False,
)

tone_mapping_pass = gfx.renderers.wgpu.ToneMappingPass()

# Add bloom pass to renderer's effect passes
hdr_renderer.effect_passes = [bloom_pass, tone_mapping_pass]
ldr_renderer.effect_passes = [bloom_pass]

# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img,
    dim=2,
    size=(cube_size, cube_size, 6),
    generate_mipmaps=True,
    colorspace=gfx.ColorSpace.srgb,
)

# Apply env map to skybox
background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_tex))
scene.add(background)

scene.environment = env_tex

# Load meshes
# Note that this lights the helmet already
gltf_path = model_dir / "DamagedHelmet" / "glTF" / "DamagedHelmet.gltf"

gltf = gfx.load_gltf(gltf_path)
# gfx.print_scene_graph(gltf.scene) # Uncomment to see the tree structure

scene.add(gltf.scene)

# Add extra light more or less where the sun seems to be in the skybox
light = gfx.SpotLight(color="#444")
light.local.position = (-500, 1000, -1000)
scene.add(light)

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 800 / 480)
camera.show_object(gltf.scene, view_dir=(1.8, -0.6, -2.7))
controller = gfx.OrbitController(camera, register_events=hdr_renderer)


hdr_renderer.auto_clear_output = False
clock = gfx.Clock()


def animate():
    ratio = 0.5 + 0.4 * np.sin(clock.get_elapsed_time())

    w, h = canvas.get_logical_size()

    ldr_renderer.scissor_rect = (0, 0, int(w * ratio), h)
    hdr_renderer.scissor_rect = (int(w * ratio), 0, int(w * (1 - ratio)), h)

    ldr_renderer.render(scene, camera)
    hdr_renderer.render(scene, camera)

    canvas.request_draw()


if __name__ == "__main__":
    ldr_renderer.request_draw(animate)
    loop.run()
