//
// RearWideCameraView.swift
// case4visionmagnifier
//
// Created by Paul Ruvolo on 8/13/25.

import SwiftUI
import AVFoundation
import UIKit
import Combine
import CoreGraphics
import Accelerate
import simd
import CoreMotion
import CoreImage
import Metal
import MetalKit

final class WarpRenderer {
    // MARK: - Metal / CI
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState
    let sampler: MTLSamplerState
    // Off-screen textures (reused & resized as needed)
    private var dstScratchTex: MTLTexture? // render target for warped output
    // (If you have a CVPixelBuffer, also keep a CVMetalTextureCache)

    struct WarpUniforms { var M: simd_float3x3; var oobAlpha: Float }

    init?(device: MTLDevice) {
        guard let q = device.makeCommandQueue() else { return nil }
        self.device = device
        queue = q

        // Pipeline (matches the shader names we discussed earlier)
        let lib = device.makeDefaultLibrary()!
        let vfn = lib.makeFunction(name: "vs_fullscreen_triangle")!
        let ffn = lib.makeFunction(name: "fs_warp")!
        let p = MTLRenderPipelineDescriptor()
        p.vertexFunction = vfn
        p.fragmentFunction = ffn
        p.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeline = try! device.makeRenderPipelineState(descriptor: p)

        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        sampler = device.makeSamplerState(descriptor: sd)!
    }

    /// Build a translation transform as a 3x3 matrix (operates on homogeneous coordinates)
    /// - Parameters:
    ///   - tx: x translation to apply
    ///   - ty: y translation to apply
    /// - Returns: a 3x3 matrix that applies the transform
    private func T(_ tx: Float, _ ty: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(1,0,0), SIMD3(0,1,0), SIMD3(tx,ty,1)))
    }
    
    /// Build a scale transform as a 3x3 matrix (operates on homogeneous coordinates)
    /// - Parameters:
    ///   - s: the scale factor to apply
    /// - Returns: a 3x3 matrix that applies the transform
    private func S(_ s: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(s,0,0), SIMD3(0,s,0), SIMD3(0,0,1)))
    }
    
    /// Convert from pixels to normalized values (0,1)
    /// - Parameters:
    ///   - W: the width of the image
    ///   - H: the height of the image
    /// - Returns: a 3x3 transform that converts to normalized values
    private func N_pix2norm(_ W: Float, _ H: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(1/W,0,0), SIMD3(0,1/H,0), SIMD3(0,0,1)))
    }
    
    /// Convert from normalized values (0,1) to pxels
    /// - Parameters:
    ///   - W: the width of the image
    ///   - H: the height of the image
    /// - Returns: a 3x3 transform that converts to pixel values
    private func N_norm2pix(_ W: Float, _ H: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(W,0,0), SIMD3(0,H,0), SIMD3(0,0,1)))
    }
    
    /// Use the given transform (usually a homography) to project the pixel coordinate x, y
    /// - Parameters:
    ///   - H: the projective transform (e.g., a homography)
    ///   - x: the x pixel coordinate
    ///   - y: the y pixel coordinate
    /// - Returns: the projected pixel coordinates
    private func project(_ H: simd_float3x3, _ x: Float, _ y: Float) -> SIMD2<Float> {
        let v = SIMD3<Float>(x,y,1)
        let w = H * v
        return SIMD2(w.x/w.z, w.y/w.z)
    }
    
    /// Computes the pixel area of a quadrilateral from its vertices.  A counterclockwise ordering of the vertices is assumed.
    /// - Parameter p: the pixel coordinates of the vertices
    /// - Returns: the area in pixels
    private func quadArea(_ p: [SIMD2<Float>]) -> Float {
        precondition(p.count == 4)
        let v = p + [p[0]]
        var a: Float = 0
        for i in 0..<4 { a += 0.5 * (v[i].x - v[i+1].x) * (v[i].y + v[i+1].y) }
        return abs(a)
    }

    private func buildDestToSourceMatrix(H_srcToDst: simd_float3x3,
                                         width W: Int, height Hgt: Int) -> simd_float3x3 {
        let Wf = Float(W), Hf = Float(Hgt)
        let center = SIMD2(Wf*0.5, Hf*0.5)
        let Hinv = H_srcToDst.inverse
        // scale based on maintaining area of the warped quadrilateral around the center pixel
        let centerOffset = Float(5.0)
        // Raw projected quad around center
        let tl = project(Hinv, Wf*0.5-centerOffset,    Hf*0.5-centerOffset)
        let tr = project(Hinv, Wf*0.5+centerOffset,  Hf*0.5-centerOffset)
        let bl = project(Hinv, Wf*0.5-centerOffset,   Hf*0.5+centerOffset)
        let br = project(Hinv, Wf*0.5+centerOffset,  Hf*0.5+centerOffset)

        // Center alignment
        let srcCenterInDst = project(Hinv, Wf*0.5, Hf*0.5)
        let offset = center - srcCenterInDst

        // Area matching
        let rawArea = quadArea([tl, bl, br, tr])
        let srcArea = centerOffset * centerOffset
        let s = 1.0/(sqrt(max(1e-8, srcArea / max(1e-8, rawArea))))

        // Pre-warp dest transform A (pixels)
        let A = T(offset.x, offset.y) * T(center.x, center.y) * S(s) * T(-center.x, -center.y)

        // Norm-space matrix M = Nsrc * Hinv * A * (Ndst^-1)
        let Nsrc = N_pix2norm(Wf, Hf)
        let NdstInv = N_norm2pix(Wf, Hf)
        return Nsrc * Hinv * A * NdstInv
    }

    // MARK: - Public: call me each frame
    func processFrame(srcTex: MTLTexture, H: simd_float3x3)->MTLTexture? {
        let dstW = Int(srcTex.width)*CameraPreview.Coordinator.destinationTextureScaleFactor
        let dstH = Int(srcTex.height)*CameraPreview.Coordinator.destinationTextureScaleFactor

        // Destination texture (warp render target)
        if dstScratchTex == nil ||
          dstScratchTex!.width != dstW || dstScratchTex!.height != dstH {
          let td = MTLTextureDescriptor.texture2DDescriptor(
              pixelFormat: .bgra8Unorm,
              width: dstW, height: dstH, mipmapped: false)
          td.usage = [.renderTarget, .shaderRead]  // we’ll read it as a CIImage
          td.storageMode = .private
          dstScratchTex = device.makeTexture(descriptor: td)
        }
        let dstTex = dstScratchTex!
        
        let cmd = queue.makeCommandBuffer()!

        // 2) Warp pass into dstTex
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = dstTex
        rp.colorAttachments[0].loadAction  = .clear
        rp.colorAttachments[0].storeAction = .store
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)

        let enc = cmd.makeRenderCommandEncoder(descriptor: rp)!
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentTexture(srcTex, index: 0)
        enc.setFragmentSamplerState(sampler, index: 0)

        var U = WarpUniforms(
            M: buildDestToSourceMatrix(H_srcToDst: H, width: dstW, height: dstH),
            oobAlpha: 0.0
        )
        enc.setFragmentBytes(&U, length: MemoryLayout<WarpUniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted() // ensure the texture is ready to read

        return dstTex
    }
}

enum FilterMode: String, CaseIterable, Identifiable {
    case none = "None"
    case blackOnWhite = "Black on white"
    case whiteOnBlack = "White on black"
    case yellowOnBlack = "Yellow on black"
    var id: Self { self }
}


// MARK: - SwiftUI Camera View
struct RearWideCameraView: View {
    @StateObject private var model = CameraModel()
    @State private var filterMode: FilterMode = .none
    @State private var isFrozen = false
    @State private var includeGuides = false
    @State private var showPicker = false
    @StateObject private var torch = TorchMonitor()
    @AppStorage("minimumMagnification") private var minimumMagnification: Double = 3
    @Environment(\.scenePhase) private var scenePhase
    
    private static let sfSymbolSize: CGFloat = 60
    
    private var buttonFGColor: Color {
        switch filterMode {
        case .blackOnWhite:
            return .black
        case .whiteOnBlack:
            return .white
        case .yellowOnBlack:
            return .white
        case .none:
            return .black
        }
    }
    
    private var buttonBGColor: Color {
        switch filterMode {
        case .blackOnWhite:
            return .white
        case .whiteOnBlack:
            return .black
        case .yellowOnBlack:
            return .black
        case .none:
            return .white
        }
    }
    
    var body: some View {
        ZStack {
            if let session = model.session {
                CameraPreview(session: session, isFrozen: $isFrozen, filterMode: $filterMode, minimumMagnification: $minimumMagnification)
                    .ignoresSafeArea()
                    .onAppear {
                        model.start()
                        if let device = model.device {
                            torch.attach(device: device)
                        }
                    }
                    .onDisappear { model.stop() }
                VStack {
                    HStack {
                        Button(action: {
                            isFrozen.toggle()
                        }) {
                            if isFrozen {
                                Image(systemName: "snowflake.slash")
                                    .font(.system(size: Self.sfSymbolSize))
                                    .foregroundColor(buttonFGColor)
                            } else {
                                Image(systemName: "snowflake")
                                    .font(.system(size: Self.sfSymbolSize))
                                    .foregroundColor(buttonFGColor)
                            }
                        }
                        .padding(5)
                        .background(
                            Circle().fill(buttonBGColor) // solid circular background
                        )
                        Spacer()
                        Button(action: {
                            switch filterMode {
                            case .none:
                                filterMode = .blackOnWhite
                            case .blackOnWhite:
                                filterMode = .whiteOnBlack
                            case .whiteOnBlack:
                                filterMode = .yellowOnBlack
                            case .yellowOnBlack:
                                filterMode = .none
                            }
                        }) {
                            Image(systemName: "camera.filters")
                                .font(.system(size: Self.sfSymbolSize))
                                .foregroundColor(buttonFGColor)
                        }
                        .padding(5)
                        .background(
                            Circle().fill(buttonBGColor) // solid circular background
                        )
                    }
                    if includeGuides {
                        Rectangle()
                            .fill(Color.red)
                            .frame(maxWidth: .infinity)
                            .frame(height: 10)
                            .ignoresSafeArea()
                    }
                    Spacer()
                    if includeGuides {
                        Rectangle()
                            .fill(Color.red)
                            .frame(maxWidth: .infinity)
                            .frame(height: 10)
                            .ignoresSafeArea()
                    }
                    HStack {
                        Button(action: {
                            guard let videoInput = session.inputs
                                .compactMap({ $0 as? AVCaptureDeviceInput })
                                .first(where: { $0.device.hasMediaType(.video) }),
                                  videoInput.device.hasTorch else {
                                return
                            }
                            let device = videoInput.device
                            if device.torchMode == .off {
                                do {
                                    try device.lockForConfiguration()
                                    try device.setTorchModeOn(level: 1.0)
                                    device.unlockForConfiguration()
                                } catch {
                                    device.unlockForConfiguration()
                                    print("torch error \(error)")
                                }
                            } else if device.torchMode == .on {
                                do {
                                    try device.lockForConfiguration()
                                    device.torchMode = .off
                                    device.unlockForConfiguration()
                                } catch {
                                    device.unlockForConfiguration()
                                    print("torch error \(error)")
                                }
                            }
                        }) {
                            if torch.isOn {
                                Image(systemName: "flashlight.slash.circle")
                                    .font(.system(size: Self.sfSymbolSize))
                                    .foregroundColor(buttonFGColor)
                            } else {
                                Image(systemName: "flashlight.on.circle")
                                    .font(.system(size: Self.sfSymbolSize))
                                    .foregroundColor(buttonFGColor)
                            }
                        }
                        .padding(5)
                        .background(
                            Circle().fill(buttonBGColor) // solid circular background
                        )
                        Spacer()
                        Button(action: {
                            includeGuides.toggle()
                        }) {
                            Image(systemName: "equal.circle")
                                .font(.system(size: Self.sfSymbolSize))
                                .foregroundColor(buttonFGColor)
                        }
                        .padding(5)
                        .background(
                            Circle().fill(buttonBGColor) // solid circular background
                        )
                    }
                }
                .padding([.top], 15) // space from edges
               
            } else {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Preparing camera…")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } .task {
                    await model.configure()
                }
            }
        }
        .alert(item: $model.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("OK")))
        }.onChange(of: scenePhase) { (oldPhase, newPhase) in
            // When returning from Settings app, values are synced.
            // No manual synchronize needed; this ensures re-render happens.
            if newPhase == .active { _ = minimumMagnification } // touch to trigger view update if needed
        }
    }
}

final class PreviewMetalView: MTKView {
    private let commandQueue: MTLCommandQueue
    private var srcTexture: MTLTexture?
    var clearTexture: MTLTexture?  // black texture sized to current drawable

    /// Mimic CALayer.contentsGravity behavior
    var contentsGravity: CALayerContentsGravity = .resizeAspectFill {
        didSet { setNeedsDisplay() }
    }

    override init(frame: CGRect = .zero, device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        let device = device ?? MTLCreateSystemDefaultDevice()!
        self.commandQueue = device.makeCommandQueue()!
        super.init(frame: frame, device: device)
        isPaused = true
        enableSetNeedsDisplay = true
        framebufferOnly = false
        isOpaque = true
        colorPixelFormat = .bgra8Unorm
        autoResizeDrawable = false
    }

    required init(coder: NSCoder) {
        let device = MTLCreateSystemDefaultDevice()!
        self.commandQueue = device.makeCommandQueue()!
        super.init(coder: coder)
        self.device = device
        isPaused = true
        enableSetNeedsDisplay = true
        framebufferOnly = false
        isOpaque = true
        colorPixelFormat = .bgra8Unorm
        autoResizeDrawable = false
    }

    // Keep drawableSize = view's pixel size to avoid CAMetalLayer stretching.
    override func layoutSubviews() {
        super.layoutSubviews()
        drawableSize = CGSize(width: bounds.width * contentScaleFactor,
                              height: bounds.height * contentScaleFactor)
        clearTexture = nil // force re-make at new size
    }

    /// Public entry: just stash and trigger a draw
    func draw(texture: MTLTexture) {
        self.srcTexture = texture
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let drawable = currentDrawable,
              let src = srcTexture,
              let cmd = commandQueue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else { return }

        // 1) Clear the drawable using a black texture via blit
        ensureClearTexture(for: drawable.texture)
        if let clearTex = clearTexture {
            let full = MTLSize(width: drawable.texture.width, height: drawable.texture.height, depth: 1)
            blit.copy(from: clearTex,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: .init(x: 0, y: 0, z: 0),
                      sourceSize: full,
                      to: drawable.texture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: .init(x: 0, y: 0, z: 0))
        }

        // 2) Compute copy regions for aspect-fit/fill (no scaling)
        let dstW = drawable.texture.width
        let dstH = drawable.texture.height
        let srcW = src.width
        let srcH = src.height

        let regions = computeCopyRegions(srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH, gravity: contentsGravity)

        // 3) Blit the selected source region into the centered destination origin (same size)
        if regions.size.width > 0 && regions.size.height > 0 {
            blit.copy(from: src,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: regions.srcOrigin,
                      sourceSize: regions.size,
                      to: drawable.texture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: regions.dstOrigin)
        }

        blit.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }

    // MARK: - Helpers

    private func ensureClearTexture(for like: MTLTexture) {
        guard clearTexture == nil ||
              clearTexture!.width  != like.width ||
              clearTexture!.height != like.height else { return }

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: like.pixelFormat,
                                                            width: like.width,
                                                            height: like.height,
                                                            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite] // generous; read/write not strictly needed
        guard let tex = device?.makeTexture(descriptor: desc) else { return }

        // Fill with zeros on CPU (bgra8Unorm -> black/opaque=0,0,0,0; if you prefer 1.0 alpha, still 0 here is fine for background)
        let bytesPerPixel = 4
        let rowBytes = like.width * bytesPerPixel
        let zeroes = [UInt8](repeating: 0, count: rowBytes * like.height)
        zeroes.withUnsafeBytes { ptr in
            let region = MTLRegionMake2D(0, 0, like.width, like.height)
            tex.replace(region: region, mipmapLevel: 0, withBytes: ptr.baseAddress!, bytesPerRow: rowBytes)
        }
        clearTexture = tex
    }

    private struct CopyRegions {
        let srcOrigin: MTLOrigin
        let dstOrigin: MTLOrigin
        let size: MTLSize
    }

    /// Compute source crop (no scaling) and destination origin for aspect-fit/fill.
    private func computeCopyRegions(srcW: Int, srcH: Int, dstW: Int, dstH: Int,
                                    gravity: CALayerContentsGravity) -> CopyRegions
    {
        let srcAR = Double(srcW) / Double(srcH)
        let dstAR = Double(dstW) / Double(dstH)

        // Start with defaults
        var copyW = min(srcW, dstW)
        var copyH = min(srcH, dstH)
        var srcX = 0
        var srcY = 0
        var dstX = (dstW - copyW) / 2
        var dstY = (dstH - copyH) / 2

        switch gravity {
        case .resizeAspectFill:
            // Fill the destination: choose a source crop with the destination's aspect ratio and (ideally) the destination size.
            // If source is at least as large as destination in both axes, we can crop source to exactly dst size.
            if srcW >= dstW && srcH >= dstH {
                // Crop source to dst aspect, then take a dst-sized window from that crop.
                // Compute maximal crop with dst aspect inside source.
                // Option A: width-limited crop
                var cropW = srcW
                var cropH = Int((Double(cropW) / dstAR).rounded(.toNearestOrAwayFromZero))
                if cropH > srcH {
                    print("height is the limiter")

                    // Height is the limiter
                    cropH = srcH
                    cropW = Int((Double(cropH) * dstAR).rounded(.toNearestOrAwayFromZero))
                }
                // Now from that crop, take a dst-sized window centered (no scaling)
                // Because src >= dst in both axes, it's safe to take dst size directly.
                copyW = dstW
                copyH = dstH
                srcX  = (srcW - copyW) / 2
                srcY  = (srcH - copyH) / 2
                dstX  = 0
                dstY  = 0
                print("copyW \(copyW) copyH \(copyH)")
            } else {
                // Source too small to truly fill without scaling; fall back to centered 1:1 copy.
                // Keep aspect by limiting to the smaller limiting dimension relative to destination.
                if dstAR > srcAR {
                    // Destination wider: height-limited by src
                    copyH = min(srcH, dstH)
                    copyW = min(Int((Double(copyH) * srcAR).rounded(.toNearestOrAwayFromZero)), dstW)
                } else {
                    // Destination taller: width-limited by src
                    copyW = min(srcW, dstW)
                    copyH = min(Int((Double(copyW) / srcAR).rounded(.toNearestOrAwayFromZero)), dstH)
                }
                srcX = (srcW - copyW) / 2
                srcY = (srcH - copyH) / 2
                dstX = (dstW - copyW) / 2
                dstY = (dstH - copyH) / 2
            }

        case .resizeAspect:
            // Fit inside destination: choose a destination box with source aspect (no scaling), center it.
            if dstAR > srcAR {
                // Destination wider than source aspect -> height limits
                copyH = min(srcH, dstH)
                copyW = min(Int((Double(copyH) * srcAR).rounded(.toNearestOrAwayFromZero)), dstW)
            } else {
                // Destination taller (or equal) -> width limits
                copyW = min(srcW, dstW)
                copyH = min(Int((Double(copyW) / srcAR).rounded(.toNearestOrAwayFromZero)), dstH)
            }
            srcX = (srcW - copyW) / 2  // center-crop if source larger
            srcY = (srcH - copyH) / 2
            dstX = (dstW - copyW) / 2  // center in destination (letterbox/pillarbox)
            dstY = (dstH - copyH) / 2

        default:
            // .resize (stretch) or others: copy the largest common sub-rect 1:1, centered.
            copyW = min(srcW, dstW)
            copyH = min(srcH, dstH)
            srcX = (srcW - copyW) / 2
            srcY = (srcH - copyH) / 2
            dstX = (dstW - copyW) / 2
            dstY = (dstH - copyH) / 2
        }

        // Clamp sanity
        copyW = max(0, min(copyW, min(srcW, dstW)))
        copyH = max(0, min(copyH, min(srcH, dstH)))

        return CopyRegions(
            srcOrigin: MTLOrigin(x: srcX, y: srcY, z: 0),
            dstOrigin: MTLOrigin(x: dstX, y: dstY, z: 0),
            size: MTLSize(width: copyW, height: copyH, depth: 1)
        )
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    @Binding var isFrozen: Bool
    @Binding var filterMode: FilterMode
    @Binding var minimumMagnification: Double

    var minZoom: CGFloat = 1.0
    var maxZoom: CGFloat = 10.0
    
    func makeCoordinator() -> Coordinator {
        Coordinator(session: session)
    }
    func makeUIView(context: Context) -> UIScrollView {
        let scroll = UIScrollView()
        scroll.minimumZoomScale = minZoom
        scroll.maximumZoomScale = maxZoom
        scroll.bouncesZoom = true
        scroll.showsVerticalScrollIndicator = false
        scroll.showsHorizontalScrollIndicator = false
        scroll.backgroundColor = .black
        scroll.delegate = context.coordinator
        scroll.isScrollEnabled = false // Container so both preview and overlay zoom together
        let container = UIView()
        container.frame = scroll.bounds
        container.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        scroll.addSubview(container)
        // Live preview
        let preview = PreviewMetalView()
        preview.frame = container.bounds
        preview.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        container.addSubview(preview)
        // Frozen overlay (hidden by default)
        let overlay = UIImageView(frame: container.bounds)
        overlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        overlay.contentMode = .scaleAspectFill
        overlay.isHidden = true
        container.addSubview(overlay)
        // Keep references
        context.coordinator.scrollView = scroll
        context.coordinator.container = container
        context.coordinator.preview = preview
        context.coordinator.overlay = overlay
        // Ensure we have a video output to grab latest frames for freezing
        context.coordinator.ensureVideoOutputOnSession()
        scroll.contentSize = container.bounds.size
        context.coordinator.startMotion() // start IMU
        // Initial recenter after the first layout pass
        DispatchQueue.main.async {
            context.coordinator.centerContent()
            context.coordinator.lockContentOffsetToCenter()
        }
        return scroll
    }
    
    func updateUIView(_ scroll: UIScrollView, context: Context) {
        context.coordinator.applyMinimumMagnification(minimumMagnification)
        context.coordinator.applyFreeze(isFrozen)
        context.coordinator.setFilterMode(filterMode)
        context.coordinator.centerContent()
    }
    
    // MARK: - Coordinator
    
    final class Coordinator: NSObject, UIScrollViewDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
        private weak var session: AVCaptureSession?
        weak var scrollView: UIScrollView?
        weak var container: UIView?
        weak var preview: PreviewMetalView?
        weak var overlay: UIImageView?
        private var filterMode: FilterMode = .none
        private let sampleQueue = DispatchQueue(label: "freeze.preview.sample")
        private let motion = CMMotionManager()
        private(set) var gravity: CMAcceleration?
        // camera intrinsics (pixels)
        private var K: simd_float3x3?
        private var processingFrame = false
        private var doPerspectiveCorrection = true
        var renderer: WarpRenderer?
        var thresholder: AdaptiveThreshColorized?
        private var frameCountDown = 0
        static let framesToDrop = 1
        let device = MTLCreateSystemDefaultDevice()!
        /// this flag communicates to the renderer and warper that we should prepare a UIImage from the result (this is a costly operation that shouldn't be done every frame)
        var prepareToFreeze = false
        var isFrozen = false
        private var mtlBridge = PixelBufferMetalBridge()
        static let destinationTextureScaleFactor = 1
        
        /// Start motion updates so we can access the gravity vector
        func startMotion() {
            guard motion.isDeviceMotionAvailable else {
                return
            }
            motion.deviceMotionUpdateInterval = 1/60
            motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] dm, _ in
                guard let self, let dm else {
                    return
                }
                self.gravity = dm.gravity
            }
        }
        
        typealias vImagePoint = (Float, Float)
        
        func convertCIImageToCGImage(inputImage: CIImage) -> CGImage? {
            let context = CIContext(options: nil)
            if let cgImage = context.createCGImage(inputImage, from: inputImage.extent) {
                return cgImage
            }
            return nil
        }
        
        func applyMinimumMagnification(_ minimumMag: Double) {
            scrollView?.minimumZoomScale = minimumMag
        }
        
        /// Pulls the camera intrinsics from the pixel buffer.  The intrinsics are needed to compute the
        /// homography that corrects for the pitch and roll of the phone.
        /// - Parameter pixelBuffer: the image that is being perspective corrected (e.g., from the camera feed)
        private func updateIntrinsicsIfNeeded(from pixelBuffer: CVPixelBuffer) {
            guard K == nil else {
                return
            }
            // Try CVImageBuffer attachment first
            if let att = CMGetAttachment(pixelBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix as CFString, attachmentModeOut: nil) {
                if let data = att as? Data, data.count >= MemoryLayout<Float>.size * 9 { let f = data.withUnsafeBytes {
                        $0.bindMemory(to: Float.self)
                    }
                    // Row-major 3x3
                    K = simd_float3x3(rows: [ SIMD3(f[0], f[1], f[2]), SIMD3(f[3], f[4], f[5]), SIMD3(f[6], f[7], f[8]) ])
                    return
                }
            }
            // Fallback: approximate from FOV + size
            let w = Float(CVPixelBufferGetWidth(pixelBuffer))
            let h = Float(CVPixelBufferGetHeight(pixelBuffer))
            let aspect = w / h
            if let device = (session?.inputs.first as? AVCaptureDeviceInput)?.device {
                let diagFOVdeg = Float(device.activeFormat.videoFieldOfView)
                // diagonal FOV
                let td = tanf(0.5 * diagFOVdeg * .pi / 180)
                // tan(diagonal/2)
                // Solve for horizontal/vertical half-FOV from diagonal + aspect:
                // td^2 = tan(h/2)^2 + tan(v/2)^2, with tan(h/2) = aspect * tan(v/2)
                let tv = td / sqrtf(aspect * aspect + 1)
                let th = aspect * tv
                let fx = w / (2 * th)
                let fy = h / (2 * tv)
                K = simd_float3x3(rows: [
                    SIMD3(fx, 0, w/2),
                    SIMD3(0, fy, h/2),
                    SIMD3(0, 0, 1) ])
            } else {
                // Last resort: assume a reasonable focal length
                let f = 0.9 * w
                K = simd_float3x3(rows: [ SIMD3(f, 0, w/2), SIMD3(0, f, h/2), SIMD3(0, 0, 1) ])
            }
        }
        
        func applyHomographyToImage(texture: MTLTexture, H: simd_float3x3)->MTLTexture? {
            if renderer == nil {
                renderer = WarpRenderer(device: device)
            }
            if thresholder == nil {
                thresholder = try? AdaptiveThreshColorized(device: device)
            }
            guard let renderer = renderer else {
                return nil
            }
            guard let thresholder = thresholder else {
                return nil
            }
            
            if filterMode == .none {
                return renderer.processFrame(srcTex: texture, H: H.inverse)
            } else {
                guard let thresholded = thresholder.processFrame(srcTex: texture, filterMode: filterMode) else {
                    return nil
                }
                return renderer.processFrame(srcTex: thresholded, H: H.inverse)
            }
        }
        
        /// Adjust the texture based on the gravity vector as returned by the CoreMotion.
        /// We assume a landscape right orientation for the image
        /// - Parameters:
        ///   - uiImage: the texture that wraps the pixel buffer from the camera feed
        /// - Returns: the corrected texture (nil otherwise)
        func rectifyToLevel(_ texture: MTLTexture) -> MTLTexture? {
            guard let gravity = gravity else {
                return nil
            }
            guard let K = self.K else {
                // If intrinsics missing, just return input
                return nil
            }
            let gravityVectorInCameraConventions = simd_float3(Float(gravity.y), Float(-gravity.x), Float(-gravity.z))
            let R = simd_float3x3(simd_quatf(from: simd_float3(0, 0, 1), to: gravityVectorInCameraConventions))
            var K_scaled = K
            let s = Float(Self.destinationTextureScaleFactor)
            K_scaled.columns.0.x *= s
            K_scaled.columns.1.y *= s
            K_scaled.columns.2.x *= s
            K_scaled.columns.2.y *= s
            let H = K * R * simd_inverse(K_scaled)
            // correct for perspective
            return applyHomographyToImage(texture: texture, H: H.inverse)
        }

        init(session: AVCaptureSession) {
            self.session = session
        }
        
        // Add (or reuse) a VideoDataOutput so we can grab the latest frame as UIImage
        func ensureVideoOutputOnSession() {
            guard let session = session else {
                return
            }
            // Reuse if one exists
            if session.outputs.contains(where: { $0 is AVCaptureVideoDataOutput }) {
                if let out = session.outputs.compactMap({ $0 as? AVCaptureVideoDataOutput }).first {
                    out.setSampleBufferDelegate(self, queue: sampleQueue)
                }
                return
            }
            let out = AVCaptureVideoDataOutput()
            out.alwaysDiscardsLateVideoFrames = true
            // we neeed kCVPixelFormatType_32BGRA to make our handling of the MTLTexture easier
            out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferMetalCompatibilityKey as String: true]

            if session.canAddOutput(out) {
                session.addOutput(out)
            }
            out.setSampleBufferDelegate(self, queue: sampleQueue)
            // Match orientation to preview
            if let conn = out.connection(with: .video) {
                if #available(iOS 17.0, *) {
                    if conn.isVideoRotationAngleSupported(90) {
                        conn.videoRotationAngle = 90
                    }
                } else if conn.isVideoOrientationSupported {
                    conn.videoOrientation = .landscapeRight
                }
            }
        }
        
        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer:
                        CMSampleBuffer, from connection: AVCaptureConnection) {
            let frameStart = Date()
            // We are careful not to process frames too slowly, so we discard frames if we are still processing a previous frame
            guard !processingFrame else {
                return
            }
            frameCountDown -= 1
            guard frameCountDown <= 0 else {
                // drop this frame
                return
            }
            frameCountDown = 1 + Self.framesToDrop
            processingFrame = true
            DispatchQueue.global(qos: .userInteractive).async { [self] in
                guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                    self.processingFrame = false
                    return
                }
                self.updateIntrinsicsIfNeeded(from: pb)
                if let texture = mtlBridge.makeTexture(from: pb), let leveled = self.rectifyToLevel(texture) {
                    DispatchQueue.main.async {
                        self.preview?.draw(texture: leveled)
                        if self.prepareToFreeze, let displayedTexture = self.preview?.currentDrawable?.texture {
                            // this pipeline is very slow, so only do it when we are about to freeze
                            let leveledCI = CIImage(
                                mtlTexture: displayedTexture,
                                options: [.colorSpace: CGColorSpaceCreateDeviceRGB()]
                            )!.oriented(.rightMirrored)
                            if let latestImage = self.convertCIImageToCGImage(inputImage: leveledCI) {
                                DispatchQueue.main.async {
                                    self.overlay?.image = UIImage(cgImage: latestImage, scale: 1.0, orientation: .right)
                                    self.overlay?.isHidden = false
                                    self.preview?.isHidden = true
                                }
                            }
                            self.prepareToFreeze = false
                        }
                    }
                }
                print("total frame processing time \(Date().timeIntervalSince(frameStart))")
                self.processingFrame = false
            }
        }
        
        func setFilterMode(_ filterMode: FilterMode) {
            self.filterMode = filterMode
        }
        
        // Toggle frozen state
        func applyFreeze(_ frozen: Bool) {
            guard let overlay, let preview else {
                return
            }
            if !frozen {
                overlay.isHidden = true
                preview.isHidden = false
            }
            if !isFrozen && frozen {
                prepareToFreeze = true
            }
            isFrozen = frozen
            // Recenter after the visibility change, even if no zoom yet.
            DispatchQueue.main.async { [weak self] in
                guard let self = self, let scroll = self.scrollView else {
                    return
                }
                scroll.layoutIfNeeded()
                self.centerContent()
                self.lockContentOffsetToCenter()
                let z = scroll.zoomScale
                scroll.setZoomScale(z * 1.0001, animated: false)
                scroll.setZoomScale(z, animated: false)
                self.centerContent()
                self.lockContentOffsetToCenter()
            }
        }

        // MARK: UIScrollViewDelegate
        func viewForZooming(in scrollView: UIScrollView) -> UIView? {
            container
        }
        func scrollViewDidZoom(_ scrollView: UIScrollView) {
            centerContent()
            lockContentOffsetToCenter()
        }
        // Center the zoomed content using insets (works when content is smaller than bounds)
        func centerContent() {
            guard let scroll = scrollView else {
                return
            }
            // If you rely on contentSize == container.bounds.size this stays correct.
            let contentSize = scroll.contentSize
            let dx = max(0, (scroll.bounds.width - contentSize.width) / 2)
            let dy = max(0, (scroll.bounds.height - contentSize.height) / 2)
            scroll.contentInset = UIEdgeInsets(top: dy, left: dx, bottom: dy, right: dx)
        }
        // Keep the visual center fixed; prevents any drift while zooming
        func lockContentOffsetToCenter() {
            guard let scroll = scrollView else {
                return
            }
            let contentSize = scroll.contentSize
            let target = CGPoint( x: max(0, (contentSize.width - scroll.bounds.width ) / 2) - scroll.adjustedContentInset.left, y: max(0, (contentSize.height - scroll.bounds.height) / 2) - scroll.adjustedContentInset.top )
            if scroll.contentOffset != target {
                scroll.setContentOffset(target, animated: false)
            }
        }
    }
}

// MARK: - Camera Model
@MainActor
final class CameraModel: ObservableObject {
    @Published var session: AVCaptureSession?
    @Published var alert: CameraAlert?
    var device: AVCaptureDevice?
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    
    func configure() async {
        // Request permission
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        let granted: Bool
        switch status {
        case .authorized:
            granted = true
        case .notDetermined:
            granted = await AVCaptureDevice.requestAccess(for: .video)
        default:
            granted = false
        }
        guard granted else {
            alert = .init(title: "Camera Access Denied", message: "Enable camera access in Settings → Privacy → Camera.")
            return
        }
        // Build the session off the main thread
        await withCheckedContinuation {
            continuation in sessionQueue.async {
                do {
                    let (session, device) = try Self.makeRearWideHighResSession()
                    DispatchQueue.main.async {
                        self.session = session
                        self.device = device
                        continuation.resume()
                    }
                }
                catch {
                    DispatchQueue.main.async {
                        self.alert = .init(title: "Camera Error", message: error.localizedDescription)
                        continuation.resume()
                    }
                }
            }
        }
    }
    
    func start() {
        guard let session = session, !session.isRunning else {
            return
        }
        sessionQueue.async {
            session.startRunning()
        }
    }
    
    func stop() {
        guard let session = session, session.isRunning else {
            return
        }
        sessionQueue.async {
            session.stopRunning()
        }
    }
    
    // MARK: Session builder (rear WIDE, highest resolution)
    
    private static func makeRearWideHighResSession() throws -> (AVCaptureSession, AVCaptureDevice) {
        let session = AVCaptureSession()
        // Use inputPriority so our chosen activeFormat takes precedence over preset.
        session.sessionPreset = .inputPriority
        // Pick the BACK WIDE-ANGLE camera (NOT ultra wide).
        // Apple device types:
        // - .builtInWideAngleCamera → the standard "Wide" camera (≈ 26mm equiv)
        // - .builtInUltraWideCamera → the Ultra Wide (≈ 13mm) — we intentionally avoid this
        let discovery = AVCaptureDevice.DiscoverySession( deviceTypes: [.builtInUltraWideCamera], mediaType: .video, position: .back )
        guard let device = discovery.devices.first else {
            throw CameraError.noRearWideCamera
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraError.cannotAddInput
        }
        session.addInput(input)
        // Choose the highest-resolution format supported by the device’s rear wide camera.
        // We’ll also pick the highest max frame rate available within that format.
        let best = bestFormatAndFrameRate(for: device)
        try device.lockForConfiguration()
        device.activeFormat = best.format
        if let fps = best.maxFPS {
            let duration = CMTimeMake(value: 1, timescale: Int32(fps.rounded()))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
        }
        
        // 1) Use continuous AF (keeps adjusting on its own)
        if device.isFocusModeSupported(.continuousAutoFocus) {
            device.focusMode = .continuousAutoFocus
        }

        // 2) Aim AF at the optical center of the active sensor
        // (Normalized coordinates: (0,0)=top-left, (1,1)=bottom-right in
        // landscape-right sensor space. Center is always 0.5,0.5.)
        if device.isFocusPointOfInterestSupported {
            print("biasing towards center")
            device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
        }

        // 3) Bias AF range toward close subjects
        if device.isAutoFocusRangeRestrictionSupported {
            print("biasing near focus")
            device.autoFocusRangeRestriction = .near
        }
        if device.isExposurePointOfInterestSupported {
            // set exposure point in the upper right to avoid darkening parts of the image too much
            device.exposurePointOfInterest = CGPoint(x: 1.0, y: 0.0)
        }
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }

        // 4) No need for smooth AF
        if device.isSmoothAutoFocusSupported {
            print("disabling smooth auto focus")
            device.isSmoothAutoFocusEnabled = false
        }
        
        device.unlockForConfiguration()
        // Add a dummy output so the session can run even if you later want to add data/video outputs.
        // For preview-only, this isn’t required, but harmless.
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        // we need kCVPixelFormatType_32BGRA to make our work with MTLTextures easier
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferMetalCompatibilityKey as String: true]

        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        return (session, device)
    }
    
    /// Find the format with the largest pixel dimensions; within that, capture the highest supported max FPS.
    private static func bestFormatAndFrameRate(for device: AVCaptureDevice) -> (format: AVCaptureDevice.Format, maxFPS: Double?) {
        // Sort formats by area (width * height), then by highest supported max frame rate.
        let sorted = device.formats.sorted { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            let areaA = Int(da.width) * Int(da.height)
            let areaB = Int(db.width) * Int(db.height)
            if areaA != areaB {
                return areaA > areaB
            }
            // Tie-breaker: higher max FPS
            let maxA = a.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            let maxB = b.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            return maxA > maxB
        }
        guard let top = sorted.first else {
            return (device.activeFormat, nil)
        }
        let maxFPS = top.videoSupportedFrameRateRanges.map(\.maxFrameRate).max()
        return (top, maxFPS)
    }
}

// MARK: - Errors & Alerts
private enum CameraError: LocalizedError {
    case noRearWideCamera
    case cannotAddInput
    var errorDescription: String? {
        switch self {
        case .noRearWideCamera:
            return "No rear wide-angle camera was found on this device."
        case .cannotAddInput:
            return "Failed to add camera input to the capture session."
        }
    }
}

struct CameraAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

final class PixelBufferMetalBridge {
    private let device = MTLCreateSystemDefaultDevice()!
    private var cache: CVMetalTextureCache!

    init() {
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
    }

    /// BGRA pixel buffer → BGRA8Unorm MTLTexture
    func makeTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        var cvTex: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            cache,
            pixelBuffer,
            nil,                       // textureAttributes
            .bgra8Unorm,               // MTLTexture pixel format
            width,
            height,
            0,                         // planeIndex (0 for single-plane)
            &cvTex
        )
        guard status == kCVReturnSuccess, let cvTex, let tex = CVMetalTextureGetTexture(cvTex) else {
            return nil
        }
        return tex
    }
}
