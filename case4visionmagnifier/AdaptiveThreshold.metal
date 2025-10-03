#include <metal_stdlib>
using namespace metal;

inline float luma709(float4 rgba) {
    return rgba.r * 0.299f + rgba.g * 0.587f + rgba.b * 0.114f;
}

struct Params {
    float C;            // if src is *Unorm*, pass C/255
    uint  binaryInv;    // 0 => THRESH_BINARY, 1 => THRESH_BINARY_INV
    float4 fg;          // foreground color (r,g,b,a) in 0..1
    float4 bg;          // background color (r,g,b,a) in 0..1
};

kernel void adaptiveThresholdToBGRA(
    texture2d<float, access::read>  srcBGRA     [[texture(0)]], // bgra8Unorm
    texture2d<float, access::read>  blurredBGRA [[texture(1)]], // bgra8Unorm
    texture2d<float, access::write> dstBGRA     [[texture(2)]], // bgra8Unorm
    constant Params&                p           [[buffer(0)]],
    uint2                           gid         [[thread_position_in_grid]])
{
    if (gid.x >= dstBGRA.get_width() || gid.y >= dstBGRA.get_height()) return;

    float s  = luma709(srcBGRA.read(gid));
    float mu = luma709(blurredBGRA.read(gid));
    bool above = (s > (mu - p.C));

    // pick color; shader sees RGBA order even for BGRA textures
    float4 on  = p.fg;
    float4 off = p.bg;

    float4 out = p.binaryInv ? (above ? off : on) : (above ? on : off);
    // (optional) clamp to be safe
    out = clamp(out, 0.0, 1.0);
    dstBGRA.write(out, gid);
}
