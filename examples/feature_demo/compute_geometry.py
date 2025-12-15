"""
Compute Shader Geometry Deformation
===================================

This example demonstrates how to use compute shaders to modify the vertex positions of a geometry mesh,
creating an interactive "jelly" effect.
"""

import numpy as np
import pygfx as gfx
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from pygfx.utils.compute import ComputeShader
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

import os
from pathlib import Path

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "data"

canvas = WgpuCanvas(size=(800, 600), max_fps=-1, title="compute geometry", vsync=False)
renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(75, 800 / 600, depth_range=(0.1, 1000))

scene = gfx.Scene()

gltf_path = model_dir / "LeePerrySmith" / "LeePerrySmith.glb"

gltf = gfx.load_gltf(gltf_path)

mesh = gltf.scene.children[0]
mesh.local.scale = (0.1, 0.1, 0.1)
mesh.material = gfx.MeshNormalMaterial(pick_write=True)

geometry = mesh.geometry

scene.add(mesh)

camera.show_object(mesh)
controller = gfx.OrbitController(camera, register_events=renderer)

count = geometry.positions.nitems
positions_data = geometry.positions.data


position_base_buffer = gfx.Buffer(
    data=positions_data,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

position_storage_buffer = gfx.Buffer(
    data=positions_data,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

geometry.positions = position_storage_buffer

speed_data = np.zeros((count, 3), dtype=np.float32)
speed_buffer = gfx.Buffer(
    data=speed_data,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

params_buffer = gfx.Buffer(
    data=np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.94, 0.25, 0.22, 0.0], dtype=np.float32),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

object_world_matrix_buffer = gfx.Buffer(
    data=mesh.world.matrix.T.astype(np.float32),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

compute_wgsl = """
struct Params {
    pointer_x: f32,
    pointer_y: f32,
    pointer_z: f32,
    pointer_active: f32,
    elasticity: f32,
    damping: f32,
    brush_size: f32,
    brush_strength: f32,
    delta_time: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> base_positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> speeds: array<f32>;
@group(0) @binding(4) var<uniform> object_world_matrix: mat4x4<f32>;



@compute @workgroup_size(64)
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&positions)/3) {
        return;
    }

    let pointer_position = vec3<f32>(params.pointer_x, params.pointer_y, params.pointer_z);

    if (params.pointer_active == 1.0) {
        // convert the position to world space
        // let current_position = positions[index];
        let current_position = vec3<f32>(positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]);
        let world_pos = (object_world_matrix * vec4<f32>(current_position, 1.0)).xyz;

        // compute the distance from the pointer position to the vertex position in world space
        let dist = distance(world_pos, pointer_position);

        if dist>0.0 {
            let direction = normalize(pointer_position - world_pos);
            // apply a force based on the distance and brush size
            let power = max(params.brush_size - dist, 0.0) * params.brush_strength;
            if (power > 0.0) {
                // positions[index] += direction * power;
                let offset = direction * power;
                positions[index * 3] += offset.x;
                positions[index * 3 + 1] += offset.y;
                positions[index * 3 + 2] += offset.z;
            }
        }

    }

    // jelly effect
    // let base_pos = base_positions[index];
    // let current_pos = positions[index];

    let base_pos = vec3<f32>(base_positions[index * 3], base_positions[index * 3 + 1], base_positions[index * 3 + 2]);
    let current_pos = vec3<f32>(positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]);

    let distance_delta = distance(base_pos, current_pos);
    let force = params.elasticity * distance_delta * (base_pos - current_pos);

    // update speeds with the force applied
    speeds[index * 3] += force.x * 60 * params.delta_time;
    speeds[index * 3 + 1] += force.y * 60 * params.delta_time;
    speeds[index * 3 + 2] += force.z * 60 * params.delta_time;

    // damping effect
    let damping_factor = pow(params.damping, 60 * params.delta_time);
    speeds[index * 3] *= damping_factor;
    speeds[index * 3 + 1] *= damping_factor;
    speeds[index * 3 + 2] *= damping_factor;

    // update positions
    positions[index * 3] += speeds[index * 3] * 60 * params.delta_time;
    positions[index * 3 + 1] += speeds[index * 3 + 1] * 60 * params.delta_time;
    positions[index * 3 + 2] += speeds[index * 3 + 2] * 60 * params.delta_time;
}
"""


update_shader = ComputeShader(
    compute_wgsl,
    entry_point="update_positions",
)


update_shader.set_resource(0, params_buffer)
update_shader.set_resource(1, position_base_buffer)
update_shader.set_resource(2, position_storage_buffer)
update_shader.set_resource(3, speed_buffer)
update_shader.set_resource(4, object_world_matrix_buffer)


params_data = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.4, 0.94, 0.25, 0.22, 0.0], dtype=np.float32
)

pointer_world_point = None


def on_pointer_move(event):
    info = event.pick_info
    if info["world_object"] is not None:
        face_index = info["face_index"]
        coords = info["face_coord"]
        sub_index = np.argmax(coords)

        vertex_index = int(event.target.geometry.indices.data[face_index, sub_index])

        pos = positions_data[vertex_index]

        global pointer_world_point
        pointer_world_point = mesh.world.matrix @ np.array(
            [pos[0], pos[1], pos[2], 1.0]
        )

    else:
        pointer_world_point = None


renderer.add_event_handler(on_pointer_move, "pointer_move")

gui_renderer = ImguiRenderer(renderer.device, canvas)


def draw_ui():
    imgui.new_frame()
    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    imgui.begin("Controls", True)

    _, params_data[4] = imgui.slider_float("Elasticity", params_data[4], 0.0, 0.5)
    _, params_data[5] = imgui.slider_float("Damping", params_data[5], 0.9, 0.99)
    _, params_data[6] = imgui.slider_float("Brush Size", params_data[6], 0.1, 0.5)
    _, params_data[7] = imgui.slider_float("Brush Strength", params_data[7], 0.1, 0.3)

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


gui_renderer.set_gui(draw_ui)

clock = gfx.Clock()


def animate():
    dt = clock.get_delta()
    params_data[-1] = dt

    global pointer_world_point

    if pointer_world_point is not None:
        params_data[0] = pointer_world_point[0]
        params_data[1] = pointer_world_point[1]
        params_data[2] = pointer_world_point[2]
        params_data[3] = 1.0
        pointer_world_point = None
    else:
        params_data[3] = 0.0

    params_buffer.set_data(params_data)
    object_world_matrix_buffer.set_data(mesh.world.matrix.T.astype(np.float32))

    update_shader.dispatch((count + 63) // 64)
    renderer.render(scene, camera)
    gui_renderer.render()

    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
