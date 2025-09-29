#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 position [[position]];
    float2 destUV;   // destination normalized coords in [0,1]
};

// Fullscreen triangle; avoids a vertex buffer.
// idx: 0,1,2 produce 3 vertices that cover the screen.
vertex VSOut vs_fullscreen_triangle(uint idx [[vertex_id]])
{
    float2 pos = float2((idx == 2) ? 3.0 : -1.0,
                        (idx == 1) ? 3.0 : -1.0);
    VSOut out;
    out.position = float4(pos, 0.0, 1.0);

    // Map clip-space to [0,1] UV
    out.destUV = 0.5 * (pos + 1.0);
    return out;
}

struct Uniforms {
    float3x3 M;   // maps dest normalized coords -> source normalized coords
    float    oobAlpha; // alpha if out-of-bounds (0..1), typically 0
};

fragment float4 fs_warp(VSOut in [[stage_in]],
                        constant Uniforms& U [[buffer(0)]],
                        texture2d<float, access::sample> src [[texture(0)]],
                        sampler samp [[sampler(0)]])
{
    // Homogeneous transform: src_uv ~ M * [dest_uv, 1]
    float3 d = float3(in.destUV, 1.0);
    float3 s = U.M * d;
    float2 srcUV = s.xy / s.z;

    // If outside source, return transparent/dimmed or clamp
    if (any(srcUV < 0.0) || any(srcUV > 1.0)) {
        return float4(0.0, 0.0, 0.0, U.oobAlpha); // background fill
    }

    return src.sample(samp, srcUV);
}
