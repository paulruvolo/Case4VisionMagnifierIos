import Metal
import MetalPerformanceShaders
import CoreImage

struct Params {
    var C: Float
    var binaryInv: UInt32
    var fg: SIMD4<Float>   // r,g,b,a in 0..1 (linear)
    var bg: SIMD4<Float>
}

enum Method { case mean, gaussian }

final class AdaptiveThreshColorized {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pso: MTLComputePipelineState
    let ciContext: CIContext
    private var srcScratchTex: MTLTexture? // holds the input CIImage
    private var dstScratchTex: MTLTexture? // render target for warped output

    init(device: MTLDevice) throws {
        self.device = device
        self.queue = device.makeCommandQueue()!
        let lib = try device.makeDefaultLibrary(bundle: .main)
        let fn  = lib.makeFunction(name: "adaptiveThresholdToBGRA")!
        self.pso = try device.makeComputePipelineState(function: fn)
        ciContext = CIContext(mtlDevice: device, options: [ .useSoftwareRenderer: false ])
    }

    /// srcBGRA/dstBGRA: MTLPixelFormat.bgra8Unorm
    func run(srcBGRA: MTLTexture,
             dstBGRA: MTLTexture,
             blockSize: Int,
             C: Float,
             binaryInv: Bool = false,
             fg: SIMD4<Float>,            // 0..1 linear
             bg: SIMD4<Float>)            // 0..1 linear
    {
        precondition(srcBGRA.pixelFormat == .bgra8Unorm && dstBGRA.pixelFormat == .bgra8Unorm)
        precondition(srcBGRA.width == dstBGRA.width && srcBGRA.height == dstBGRA.height)

        let W = srcBGRA.width, H = srcBGRA.height

        guard let cmd = queue.makeCommandBuffer() else { return }

        // 1) blurred BGRA
        let blurDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: W, height: H, mipmapped: false)
        blurDesc.usage = [.shaderRead, .shaderWrite]
        guard let blurred = device.makeTexture(descriptor: blurDesc) else { return }

        // 2) blur via MPS
        let k = max(1, blockSize | 1)
        let box = MPSImageBox(device: device, kernelWidth: k, kernelHeight: k)
        box.edgeMode = .clamp
        box.encode(commandBuffer: cmd, sourceTexture: srcBGRA, destinationTexture: blurred)

        // 3) compare → colorized BGRA
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setTexture(srcBGRA, index: 0)
        enc.setTexture(blurred, index: 1)
        enc.setTexture(dstBGRA, index: 2)

        var p = Params(
            C: C / 255.0,                 // bgra8Unorm samples as 0..1
            binaryInv: binaryInv ? 1 : 0,
            fg: fg, bg: bg
        )
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)

        let w = pso.threadExecutionWidth
        let h = max(1, pso.maxTotalThreadsPerThreadgroup / w)
        enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()
    }
    
    @inline(__always)
    func rgbaBytesToLinear(_ r: UInt8, _ g: UInt8, _ b: UInt8, _ a: UInt8 = 255) -> SIMD4<Float> {
        SIMD4(Float(r)/255, Float(g)/255, Float(b)/255, Float(a)/255)
    }
    
    // MARK: - Public: called each frame
    func processFrame(srcTex: MTLTexture, filterMode: FilterMode)->MTLTexture? {
        let dstW = Int(srcTex.width)
        let dstH = Int(srcTex.height)

        // Destination texture (warp render target)
        if dstScratchTex == nil ||
          dstScratchTex!.width != dstW || dstScratchTex!.height != dstH {
          let td = MTLTextureDescriptor.texture2DDescriptor(
              pixelFormat: .bgra8Unorm,
              width: dstW, height: dstH, mipmapped: false)
            td.usage = [.renderTarget, .shaderRead, .shaderWrite]  // we’ll read it as a CIImage
          td.storageMode = .private
          dstScratchTex = device.makeTexture(descriptor: td)
        }
        let fg: SIMD4<Float>
        let bg: SIMD4<Float>
        switch filterMode {
        case .none:
            // should not get here, use black on white since we need something
            fg = rgbaBytesToLinear(0, 0, 0)
            bg = rgbaBytesToLinear(255, 255, 255)
        case .blackOnWhite:
            fg = rgbaBytesToLinear(0, 0, 0)
            bg = rgbaBytesToLinear(255, 255, 255)
        case .whiteOnBlack:
            fg = rgbaBytesToLinear(255, 255, 255)
            bg = rgbaBytesToLinear(0, 0, 0)
        case .yellowOnBlack:
            fg = rgbaBytesToLinear(255, 255, 0)
            bg = rgbaBytesToLinear(0, 0, 0)
        }
        let dstTex = dstScratchTex!
        run(srcBGRA: srcTex,
            dstBGRA: dstTex,
            blockSize: 101,
            C: 30,
            binaryInv: true,
            fg: fg,
            bg: bg)
        return dstTex
    }
}
