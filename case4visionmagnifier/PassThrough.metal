#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

struct Uniforms {
    float2 scaleUV;   // scales uv for aspect-fit/fill (e.g., (1, s) or (s, 1))
    float2 offsetUV;  // centers the image (usually (0,0) unless letterboxing)
    float2 flipUV;    // (1,1) normally; set to (1,-1) etc. to flip
};

// Fullscreen triangle (no vertex buffer)
vertex VSOut vs_fullscreen(uint vid [[vertex_id]],
                           constant Uniforms& U [[buffer(0)]])
{
    // 3 vertices covering NDC [-1,1]^2
    float2 ndc = float2( (vid == 2) ?  3.0 : -1.0,
                         (vid == 1) ?  3.0 : -1.0 );
    VSOut o;
    o.position = float4(ndc, 0, 1);

    // Map NDC -> [0,1] then apply aspect scale/offset and optional flips
    float2 uv = 0.5 * (ndc + 1.0);
    uv = (uv - 0.5) * U.scaleUV + 0.5 + U.offsetUV;
    uv = float2( (U.flipUV.x > 0 ? uv.x : (1.0 - uv.x)),
                 (U.flipUV.y > 0 ? uv.y : (1.0 - uv.y)) );
    o.uv = uv;
    return o;
}

fragment float4 fs_sample(VSOut in [[stage_in]],
                          texture2d<float> srcTex [[texture(0)]],
                          sampler s [[sampler(0)]])
{
    // Outside the [0,1] range, return transparent/black
    if (any(in.uv < 0.0) || any(in.uv > 1.0)) {
        return float4(0,0,0,1); // or 0 alpha if you prefer
    }
    return srcTex.sample(s, in.uv);
}
