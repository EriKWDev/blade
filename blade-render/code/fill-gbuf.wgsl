#include "quaternion.inc.wgsl"
#include "camera.inc.wgsl"
#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "gbuf.inc.wgsl"

// Has to match the host!
struct Vertex {
    pos: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
}
struct VertexBuffer {
    data: array<Vertex>,
}
struct IndexBuffer {
    data: array<u32>,
}
var<storage, read> vertex_buffers: binding_array<VertexBuffer>;
var<storage, read> index_buffers: binding_array<IndexBuffer>;
var textures: binding_array<texture_2d<f32>>;
var sampler_linear: sampler;
var sampler_nearest: sampler;

struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    winding: f32,
    // packed quaternion
    geometry_to_world_rotation: u32,
    geometry_to_object: mat4x3<f32>,
    prev_object_to_world: mat4x3<f32>,
    base_color_texture: u32,
    // packed color factor
    base_color_factor: u32,
    normal_texture: u32,
    normal_scale: f32,
}
var<storage, read> hit_entries: array<HitEntry>;

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> debug: DebugParams;
var acc_struct: acceleration_structure;

var out_depth: texture_storage_2d<r32float, write>;
var out_flat_normal: texture_storage_2d<rgba8snorm, write>;
var out_basis: texture_storage_2d<rgba8snorm, write>;
var out_albedo: texture_storage_2d<rgba8unorm, write>;
var out_motion: texture_storage_2d<rg8snorm, write>;
var out_debug: texture_storage_2d<rgba8unorm, write>;

fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

fn debug_raw_normal(pos: vec3<f32>, normal_raw: u32, rotation: vec4<f32>, debug_len: f32, color: u32) {
    let nw = normalize(qrot(rotation, decode_normal(normal_raw)));
    debug_line(pos, pos + debug_len * nw, color);
}

@compute @workgroup_size(8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy >= camera.target_size)) {
        return;
    }
    if (WRITE_DEBUG_IMAGE && debug.view_mode != DebugMode_Final) {
        textureStore(out_debug, global_id.xy, vec4<f32>(0.0));
    }

    var rq: ray_query;
    let ray_dir = get_ray_direction(camera, vec2<i32>(global_id.xy));
    rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_CULL_NO_OPAQUE, 0xFFu, 0.0, camera.depth, camera.position, ray_dir));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var depth = 0.0;
    var basis = vec4<f32>(0.0);
    var flat_normal = vec3<f32>(0.0);
    var albedo = vec3<f32>(1.0);
    var motion = vec2<f32>(0.0);
    let enable_debug = all(global_id.xy == debug.mouse_pos);

    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let entry = hit_entries[intersection.instance_custom_data + intersection.geometry_index];
        depth = intersection.t;

        var indices = intersection.primitive_index * 3u + vec3<u32>(0u, 1u, 2u);
        if (entry.index_buf != ~0u) {
            let iptr = &index_buffers[entry.index_buf].data;
            indices = vec3<u32>((*iptr)[indices.x], (*iptr)[indices.y], (*iptr)[indices.z]);
        }

        let vptr = &vertex_buffers[entry.vertex_buf].data;
        let vertices = array<Vertex, 3>(
            (*vptr)[indices.x],
            (*vptr)[indices.y],
            (*vptr)[indices.z],
        );

        let positions_object = entry.geometry_to_object * mat3x4(
            vec4<f32>(vertices[0].pos, 1.0), vec4<f32>(vertices[1].pos, 1.0), vec4<f32>(vertices[2].pos, 1.0)
        );
        let positions = intersection.object_to_world * mat3x4(
            vec4<f32>(positions_object[0], 1.0), vec4<f32>(positions_object[1], 1.0), vec4<f32>(positions_object[2], 1.0)
        );
        flat_normal = entry.winding * normalize(cross(positions[1].xyz - positions[0].xyz, positions[2].xyz - positions[0].xyz));

        let barycentrics = vec3<f32>(1.0 - intersection.barycentrics.x - intersection.barycentrics.y, intersection.barycentrics);
        let position_object = vec4<f32>(positions_object * barycentrics, 1.0);
        let tex_coords = mat3x2(vertices[0].tex_coords, vertices[1].tex_coords, vertices[2].tex_coords) * barycentrics;
        let normal_geo = normalize(mat3x3(decode_normal(vertices[0].normal), decode_normal(vertices[1].normal), decode_normal(vertices[2].normal)) * barycentrics);
        let tangent_geo = normalize(mat3x3(decode_normal(vertices[0].tangent), decode_normal(vertices[1].tangent), decode_normal(vertices[2].tangent)) * barycentrics);
        let bitangent_geo = normalize(cross(normal_geo, tangent_geo)) * vertices[0].bitangent_sign;

        let lod = 0.0; //TODO: this is actually complicated

        let geo_to_world_rot = normalize(unpack4x8snorm(entry.geometry_to_world_rotation));
        let tangent_space_geo = mat3x3(tangent_geo, bitangent_geo, normal_geo);
        var normal_local: vec3<f32>;
        if ((debug.texture_flags & DebugTextureFlags_NORMAL) != 0u) {
            normal_local = vec3<f32>(0.0, 0.0, 1.0); // ignore normal map
        } else {
            let raw_unorm = textureSampleLevel(textures[entry.normal_texture], sampler_linear, tex_coords, lod).xy;
            let n_xy = entry.normal_scale * (2.0 * raw_unorm - 1.0);
            normal_local = vec3<f32>(n_xy, sqrt(max(0.0, 1.0 - dot(n_xy, n_xy))));
        }
        var normal = qrot(geo_to_world_rot, tangent_space_geo * normal_local);
        basis = shortest_arc_quat(vec3<f32>(0.0, 0.0, 1.0), normalize(normal));

        let hit_position = camera.position + intersection.t * ray_dir;
        if (enable_debug) {
            debug_buf.entry.custom_index = intersection.instance_custom_data;
            debug_buf.entry.depth = intersection.t;
            debug_buf.entry.tex_coords = tex_coords;
            debug_buf.entry.base_color_texture = entry.base_color_texture;
            debug_buf.entry.normal_texture = entry.normal_texture;
            debug_buf.entry.position = hit_position;
            debug_buf.entry.flat_normal = flat_normal;
        }
        if (enable_debug && (debug.draw_flags & DebugDrawFlags_SPACE) != 0u) {
            let normal_w = 0.15 * intersection.t * qrot(geo_to_world_rot, normal_geo);
            let tangent_w = 0.05 * intersection.t * qrot(geo_to_world_rot, tangent_geo);
            let bitangent_w = 0.05 * intersection.t * qrot(geo_to_world_rot, bitangent_geo);
            debug_line(hit_position, hit_position + normal_w, 0xFF8000u);
            debug_line(hit_position - 0.5 * tangent_w, hit_position + tangent_w, 0x8080FFu);
            debug_line(hit_position - 0.5 * bitangent_w, hit_position + bitangent_w, 0x80FF80u);
        }
        if (enable_debug && (debug.draw_flags & DebugDrawFlags_GEOMETRY) != 0u) {
            let debug_len = intersection.t * 0.2;
            debug_line(positions[0].xyz, positions[1].xyz, 0x00FFFFu);
            debug_line(positions[1].xyz, positions[2].xyz, 0x00FFFFu);
            debug_line(positions[2].xyz, positions[0].xyz, 0x00FFFFu);
            let poly_center = (positions[0].xyz + positions[1].xyz + positions[2].xyz) / 3.0;
            debug_line(poly_center, poly_center + 0.2 * debug_len * flat_normal, 0xFF00FFu);
            // note: dynamic indexing into positions isn't allowed by WGSL yet
            debug_raw_normal(positions[0].xyz, vertices[0].normal, geo_to_world_rot, 0.5*debug_len, 0xFFFF00u);
            debug_raw_normal(positions[1].xyz, vertices[1].normal, geo_to_world_rot, 0.5*debug_len, 0xFFFF00u);
            debug_raw_normal(positions[2].xyz, vertices[2].normal, geo_to_world_rot, 0.5*debug_len, 0xFFFF00u);
            // draw tangent space
            debug_line(hit_position, hit_position + debug_len * qrot(basis, vec3<f32>(1.0, 0.0, 0.0)), 0x0000FFu);
            debug_line(hit_position, hit_position + debug_len * qrot(basis, vec3<f32>(0.0, 1.0, 0.0)), 0x00FF00u);
            debug_line(hit_position, hit_position + debug_len * qrot(basis, vec3<f32>(0.0, 0.0, 1.0)), 0xFF0000u);
        }

        let base_color_factor = unpack4x8unorm(entry.base_color_factor);
        if ((debug.texture_flags & DebugTextureFlags_ALBEDO) != 0u) {
            albedo = base_color_factor.xyz;
        } else {
            let base_color_sample = textureSampleLevel(textures[entry.base_color_texture], sampler_linear, tex_coords, lod);
            albedo = (base_color_factor * base_color_sample).xyz;
        }

        if (WRITE_DEBUG_IMAGE) {
            if (debug.view_mode == DebugMode_DiffuseAlbedoTexture) {
                textureStore(out_debug, global_id.xy, vec4<f32>(albedo, 0.0));
            }
            if (debug.view_mode == DebugMode_DiffuseAlbedoFactor) {
                textureStore(out_debug, global_id.xy, base_color_factor);
            }
            if (debug.view_mode == DebugMode_NormalTexture) {
                textureStore(out_debug, global_id.xy, vec4<f32>(normal_local, 0.0));
            }
            if (debug.view_mode == DebugMode_NormalScale) {
                textureStore(out_debug, global_id.xy, vec4<f32>(entry.normal_scale));
            }
            if (debug.view_mode == DebugMode_GeometryNormal) {
                textureStore(out_debug, global_id.xy, vec4<f32>(normal_geo, 0.0));
            }
            if (debug.view_mode == DebugMode_ShadingNormal) {
                textureStore(out_debug, global_id.xy, vec4<f32>(normal, 0.0));
            }
            if (debug.view_mode == DebugMode_HitConsistency) {
                let reprojected = get_projected_pixel(camera, hit_position);
                let barycentrics_pos_diff = (intersection.object_to_world * position_object).xyz - hit_position;
                let camera_projection_diff = vec2<f32>(global_id.xy) - vec2<f32>(reprojected);
                let consistency = vec4<f32>(length(barycentrics_pos_diff), length(camera_projection_diff), 0.0, 0.0);
                textureStore(out_debug, global_id.xy, consistency);
            }
        }

        let prev_position = (entry.prev_object_to_world * position_object).xyz;
        let prev_screen = get_projected_pixel_float(prev_camera, prev_position);
        //TODO: consider just storing integers here?
        //TODO: technically this "0.5" is just a waste compute on both packing and unpacking
        motion = prev_screen - vec2<f32>(global_id.xy) - 0.5;
        if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_Motion) {
            textureStore(out_debug, global_id.xy, vec4<f32>(motion * MOTION_SCALE + vec2<f32>(0.5), 0.0, 1.0));
        }
    } else {
        if (enable_debug) {
            debug_buf.entry = DebugEntry();
        }
    }

    // TODO: option to avoid writing data for the sky
    textureStore(out_depth, global_id.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
    textureStore(out_basis, global_id.xy, basis);
    textureStore(out_flat_normal, global_id.xy, vec4<f32>(flat_normal, 0.0));
    textureStore(out_albedo, global_id.xy, vec4<f32>(albedo, 0.0));
    textureStore(out_motion, global_id.xy, vec4<f32>(motion * MOTION_SCALE, 0.0, 0.0));
}
