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
import opencv2
import Metal
import MetalKit


// 3x3 helpers
func T(_ tx: Float, _ ty: Float) -> simd_float3x3 {
    simd_float3x3(columns: (
        SIMD3(1, 0, 0),
        SIMD3(0, 1, 0),
        SIMD3(tx, ty, 1)
    ))
}
func S(_ s: Float) -> simd_float3x3 {
    simd_float3x3(columns: (
        SIMD3(s, 0, 0),
        SIMD3(0, s, 0),
        SIMD3(0, 0, 1)
    ))
}
func N_pix2norm(_ W: Float, _ H: Float) -> simd_float3x3 {
    // (x_px, y_px, 1) -> (x/W, y/H, 1)
    simd_float3x3(columns: (
        SIMD3(1.0 / W, 0, 0),
        SIMD3(0, 1.0 / H, 0),
        SIMD3(0, 0, 1)
    ))
}
func N_norm2pix(_ W: Float, _ H: Float) -> simd_float3x3 {
    simd_float3x3(columns: (
        SIMD3(W, 0, 0),
        SIMD3(0, H, 0),
        SIMD3(0, 0, 1)
    ))
}

// Project pixel point through a 3x3 homography
func project(_ H: simd_float3x3, _ x: Float, _ y: Float) -> SIMD2<Float> {
    let v = SIMD3<Float>(x, y, 1)
    let w = H * v
    return SIMD2(w.x / w.z, w.y / w.z)
}

// Shoelace area (positive, in pixel^2), vertices must be ordered around the quad
func quadArea(_ p: [SIMD2<Float>]) -> Float {
    precondition(p.count == 4)
    let v = p + [p[0]]
    var a: Float = 0
    for i in 0..<4 {
        a += 0.5 * (v[i].x - v[i+1].x) * (v[i].y + v[i+1].y)
    }
    return abs(a)
}

struct WarpUniforms {
    var M: simd_float3x3      // destNorm -> srcNorm
    var oobAlpha: Float
}

func buildDestToSourceMatrix(
    H_srcToDst: simd_float3x3,
    width W: Int, height Hgt: Int
) -> simd_float3x3 {
    let Wf = Float(W), Hf = Float(Hgt)
    let center = SIMD2(Wf * 0.5, Hf * 0.5)

    // 1) Raw projected quad (dest pixel coords) by mapping src corners through H
    //    BUT to invert in shader, we need dest->src, so compute corners by Hinv
    let Hinv = H_srcToDst.inverse

    let tl = project(Hinv, 0,    0)
    let tr = project(Hinv, Wf,   0)
    let bl = project(Hinv, 0,   Hf)
    let br = project(Hinv, Wf,  Hf)

    // 2) Centering: translate so that the projected center lands at the dest center
    //    Project the source center through Hinv, then compute offset to screen center.
    let srcCenterInDst = project(Hinv, Wf * 0.5, Hf * 0.5)
    let offset = center - srcCenterInDst      // amount to add in dest pixel space

    // 3) Area matching scale about center.
    let rawArea = quadArea([tl, bl, br, tr])  // area of projected quad (pixel^2)
    let srcArea = Wf * Hf
    // Scale so areas match: s = sqrt(srcArea / rawArea)
    let s = sqrt(max(1e-8, srcArea / max(1e-8, rawArea)))

    // 4) Build A: dest-space transform applied BEFORE Hinv (i.e., to dest pixels)
    //    A = Translate(+center) * Scale(s) * Translate(-center) * Translate(offset)
    //    Order matters; we want to first nudge by offset, then scale about center.
    let A = T(offset.x, offset.y) * T(center.x, center.y) * S(s) * T(-center.x, -center.y)

    // 5) We want M_norm = Nsrc * Hinv * A * Ndst^-1
    //    (destNorm -> destPix) -> A -> (still destPix) -> Hinv -> srcPix -> (srcNorm)
    let Nsrc = N_pix2norm(Wf, Hf)
    let NdstInv = N_norm2pix(Wf, Hf)
    let M_norm = Nsrc * Hinv * A * NdstInv
    return M_norm
}

final class WarpRenderer {
    // MARK: - Metal / CI
    let device: MTLDevice
    let queue: MTLCommandQueue
    let ciContext: CIContext
    let pipeline: MTLRenderPipelineState
    let sampler: MTLSamplerState
    // Off-screen textures (reused & resized as needed)
    private var srcScratchTex: MTLTexture? // holds the input CIImage
    private var dstScratchTex: MTLTexture? // render target for warped output
    // (If you have a CVPixelBuffer, also keep a CVMetalTextureCache)

    struct WarpUniforms { var M: simd_float3x3; var oobAlpha: Float }

    init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { return nil }
        device = dev
        queue = q
        ciContext = CIContext(mtlDevice: dev, options: [ .useSoftwareRenderer: false ])

        // Pipeline (matches the shader names we discussed earlier)
        let lib = try! dev.makeDefaultLibrary()!
        let vfn = lib.makeFunction(name: "vs_fullscreen_triangle")!
        let ffn = lib.makeFunction(name: "fs_warp")!
        let p = MTLRenderPipelineDescriptor()
        p.vertexFunction = vfn
        p.fragmentFunction = ffn
        p.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeline = try! dev.makeRenderPipelineState(descriptor: p)

        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        sampler = dev.makeSamplerState(descriptor: sd)!

        //super.init()
    }

    // MARK: - The matrix builder (from the previous message)
    private func T(_ tx: Float, _ ty: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(1,0,0), SIMD3(0,1,0), SIMD3(tx,ty,1)))
    }
    private func S(_ s: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(s,0,0), SIMD3(0,s,0), SIMD3(0,0,1)))
    }
    private func N_pix2norm(_ W: Float, _ H: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(1/W,0,0), SIMD3(0,1/H,0), SIMD3(0,0,1)))
    }
    private func N_norm2pix(_ W: Float, _ H: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(W,0,0), SIMD3(0,H,0), SIMD3(0,0,1)))
    }
    private func project(_ H: simd_float3x3, _ x: Float, _ y: Float) -> SIMD2<Float> {
        let v = SIMD3<Float>(x,y,1); let w = H * v; return SIMD2(w.x/w.z, w.y/w.z)
    }
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

        // Raw projected quad (dest pixel coords)
        let tl = project(Hinv, 0,    0)
        let tr = project(Hinv, Wf,   0)
        let bl = project(Hinv, 0,   Hf)
        let br = project(Hinv, Wf,  Hf)

        // Center alignment
        let srcCenterInDst = project(Hinv, Wf*0.5, Hf*0.5)
        let offset = center - srcCenterInDst

        // Area matching
        let rawArea = quadArea([tl, bl, br, tr])
        let srcArea = Wf * Hf
        let s = 1.0/(sqrt(max(1e-8, srcArea / max(1e-8, rawArea))))

        // Pre-warp dest transform A (pixels)
        let A = T(offset.x, offset.y) * T(center.x, center.y) * S(s) * T(-center.x, -center.y)

        // Norm-space matrix M = Nsrc * Hinv * A * (Ndst^-1)
        let Nsrc = N_pix2norm(Wf, Hf)
        let NdstInv = N_norm2pix(Wf, Hf)
        return Nsrc * Hinv * A * NdstInv
    }

    // MARK: - Public: call me each frame
    func processFrame(ciImage: CIImage, H: simd_float3x3)->CIImage? {
        let dstW = Int(ciImage.extent.width)*CameraPreview.Coordinator.destinationTextureScaleFactor
        let dstH = Int(ciImage.extent.height)*CameraPreview.Coordinator.destinationTextureScaleFactor
        print(dstW, dstH)

        // 1) Make a source texture from the CIImage (GPU path, no CPU readback)
        let srcTex: MTLTexture = {
            // Reuse a texture of the right size
            if srcScratchTex == nil ||
               srcScratchTex!.width  != Int(ciImage.extent.width) ||
               srcScratchTex!.height != Int(ciImage.extent.height) {
                let td = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .bgra8Unorm,
                    width: Int(ciImage.extent.width),
                    height: Int(ciImage.extent.height),
                    mipmapped: false)
                td.usage = [.shaderRead, .shaderWrite, .renderTarget]
                srcScratchTex = device.makeTexture(descriptor: td)
            }
            return srcScratchTex!
        }()

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

        // Render the CIImage into srcTex (still on GPU)
        ciContext.render(ciImage,
                         to: srcTex,
                         commandBuffer: cmd,
                         bounds: ciImage.extent,
                         colorSpace: CGColorSpaceCreateDeviceRGB())
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

        // --- Wrap as CIImage (GPU-backed) ---
        // (If the result appears vertically flipped in your CI workflow, apply .oriented(.downMirrored).)
        let outCI = CIImage(
            mtlTexture: dstTex,
            options: [.colorSpace: CGColorSpaceCreateDeviceRGB()]
        )!.oriented(.downMirrored) // adjust/remove if you see a flip

        return outCI
    }
}

extension CGImage {
    func resize(size:CGSize) -> CGImage? {
        let width: Int = Int(size.width)
        let height: Int = Int(size.height)

        let bytesPerPixel = self.bitsPerPixel / self.bitsPerComponent
        let destBytesPerRow = width * bytesPerPixel


        guard let colorSpace = self.colorSpace else { return nil }
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: self.bitsPerComponent, bytesPerRow: destBytesPerRow, space: colorSpace, bitmapInfo: self.alphaInfo.rawValue) else { return nil }

        context.interpolationQuality = .high
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()
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
    @State private var correctPerspective = true
    @AppStorage("minimumMagnification") private var minimumMagnification: Double = 1.5
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
                CameraPreview(session: session, isFrozen: $isFrozen, filterMode: $filterMode, minimumMagnification: $minimumMagnification, doPerspectiveCorrection: $correctPerspective)
                    .ignoresSafeArea()
                    .onAppear {
                        model.start()
                        if let device = model.device {
                            print("attaching!")
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
//                        Spacer()
//                        Button(action: { correctPerspective.toggle() }){
//                            Text("Toggle Perspective Correction")
//                        }
//                        .fontWeight(.bold)
//                        .padding()
//                        .background(
//                            RoundedRectangle(cornerRadius: 12)
//                                .stroke(Color.blue, lineWidth: 5)
//                                .fill(Color.white)
//                        )
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
                    Spacer()
                    HStack {
                        Button(action: {
                            guard let videoInput = session.inputs
                                .compactMap({ $0 as? AVCaptureDeviceInput })
                                .first(where: { $0.device.hasMediaType(.video) }),
                                  videoInput.device.hasTorch else {
                                return
                                // e.g., toggle torch or adjust focus
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
//                        Spacer()
//                        Button(action: { self.settingsOpener()} ){
//                            Text("Open Settings")
//                        }
//                        .fontWeight(.bold)
//                        .padding()
//                        .background(
//                            RoundedRectangle(cornerRadius: 12)
//                                .stroke(Color.blue, lineWidth: 5)
//                                .fill(Color.white)
//                        )
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
            if includeGuides {
                GeometryReader { geo in
                    Rectangle()
                        .fill(Color.red)
                        .frame(width: geo.size.width, height: 10)
                        .position(x: geo.size.width/2, y: 110)
                    Rectangle()
                        .fill(Color.red)
                        .frame(width: geo.size.width, height: 10)
                        .position(x: geo.size.width/2, y: geo.size.height-110)
                }.ignoresSafeArea()
            }
        }
        .alert(item: $model.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("OK")))
        }.onAppear {
            AppDelegate.orientationLock = .landscapeRight
        }.onDisappear {
            AppDelegate.orientationLock = .all // restore
        }
        .onChange(of: scenePhase) { (oldPhase, newPhase) in
            // When returning from Settings app, values are synced.
            // No manual synchronize needed; this ensures re-render happens.
            if newPhase == .active { _ = minimumMagnification } // touch to trigger view update if needed
        }
    }
    private func settingsOpener(){
        if let url = URL(string: UIApplication.openSettingsURLString) {
            if UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url, options: [:], completionHandler: nil)
            }
        }
    }
}

final class PreviewView: UIView {

    // Use a plain CALayer so we can set `.contents`
    override class var layerClass: AnyClass { CALayer.self }

    private let ciContext = CIContext() // Metal-backed by default
    private let displayQueue = DispatchQueue(label: "PreviewView.display", qos: .userInitiated)

    /// Forward to underlying CALayer for convenience
    var contentsGravity: CALayerContentsGravity {
        get { layer.contentsGravity }
        set { layer.contentsGravity = newValue }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        layer.contentsGravity = .resizeAspectFill
        layer.isGeometryFlipped = false
        layer.masksToBounds = true
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        layer.contentsGravity = .resizeAspectFill
        layer.isGeometryFlipped = false
        layer.masksToBounds = true
    }

    /// Display a CIImage (fastest path when you already have CIImage)
    func display(ciImage: CIImage, oriented orientation: CGImagePropertyOrientation = .right) {
        let image = ciImage.oriented(orientation)
        renderAndSetContents(ciImage: image)
    }

    /// Display a CVPixelBuffer
    func display(pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        renderAndSetContents(ciImage: ciImage)
    }

    /// Display a CGImage (if you already converted upstream)
    func display(cgImage: CGImage) {
        // Setting .contents must occur on main; disable implicit animations to avoid flicker
        DispatchQueue.main.async {
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            self.layer.contents = cgImage
            CATransaction.commit()
        }
    }

    // MARK: - Private

    private func renderAndSetContents(ciImage: CIImage) {
        // Render off the main thread, then set .contents on the main thread.
        displayQueue.async {
            let rect = ciImage.extent
            // Use sRGB color space to avoid washed out colors
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            guard let cgImage = self.ciContext.createCGImage(ciImage, from: rect, format: .RGBA8, colorSpace: colorSpace) else { return }

            DispatchQueue.main.async {
                CATransaction.begin()
                CATransaction.setDisableActions(true)
                self.layer.contents = cgImage
                CATransaction.commit()
            }
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    @Binding var isFrozen: Bool
    @Binding var filterMode: FilterMode
    @Binding var minimumMagnification: Double
    @Binding var doPerspectiveCorrection: Bool

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
        let preview = PreviewView()
        preview.frame = container.bounds
        preview.autoresizingMask = [.flexibleWidth, .flexibleHeight]
//        preview.videoPreviewLayer.session = session
//        preview.videoPreviewLayer.videoGravity = .resizeAspectFill
//        preview.videoPreviewLayer.connection?.videoOrientation = .landscapeRight
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
        context.coordinator.applyDoPerspectiveCorrection(doPerspectiveCorrection)
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
        weak var preview: PreviewView?
        weak var overlay: UIImageView?
        private var filterMode: FilterMode = .none
        private let ciContext = CIContext()
        private let sampleQueue = DispatchQueue(label: "freeze.preview.sample")
        private var latestImage: UIImage?
        private let motion = CMMotionManager()
        private(set) var gravity: CMAcceleration?
        // camera intrinsics (pixels)
        private var K: simd_float3x3?
        private var processingFrame = false
        private let ciCtx = CIContext(options: nil)
        private var doPerspectiveCorrection = true
        var renderer: WarpRenderer?
        
        // Create once
        let device = MTLCreateSystemDefaultDevice()!
        
        let psDesc = MTLRenderPipelineDescriptor()
        let library: MTLLibrary
        let pipeline: MTLRenderPipelineState
        let sampler: MTLSamplerState
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
        
        func applyDoPerspectiveCorrection(_ perspectiveCorrection: Bool) {
            print("perspectiveCorrection \(perspectiveCorrection)")
            doPerspectiveCorrection = perspectiveCorrection
        }
        typealias vImagePoint = (Float, Float)

        private func scale(_ p: vImagePoint, around: vImagePoint, byFactor factor: Float)->vImagePoint {
            return ((p.0 - around.0)*factor + around.0, (p.1 - around.1)*factor + around.1)
        }
        
        private func calcArea(vertices: [vImagePoint])->Float {
            guard vertices.count == 4 else {
                return 0.0
            }
            var area: Float = 0.0
            let verticesAugmented = vertices + [vertices[0]]
            for i in 0..<verticesAugmented.count - 1 {
                let vi = verticesAugmented[i]
                let viplus1 = verticesAugmented[i+1]
                area += 0.5*(vi.0 - viplus1.0)*(vi.1 + viplus1.1)
            }
            return area
        }

        func makeBlackImage(matching source: CGImage) -> CGImage? {
            let width = source.width
            let height = source.height
            let bitsPerComponent = 8
            let bytesPerRow = width * 4
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            
            guard let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                return nil
            }
            
            // Fill with black
            context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            
            return context.makeImage()
        }
        
        // Per frame:
        func drawWarp(commandBuffer: MTLCommandBuffer,
                      renderPass: MTLRenderPassDescriptor,
                      srcTex: MTLTexture,
                      H_srcToDst: simd_float3x3,
                      dstW: Int, dstH: Int)
        {
            let enc = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass)!
            enc.setRenderPipelineState(pipeline)
            enc.setFragmentTexture(srcTex, index: 0)
            enc.setFragmentSamplerState(sampler, index: 0)

            var U = WarpUniforms(
                M: buildDestToSourceMatrix(H_srcToDst: H_srcToDst, width: dstW, height: dstH),
                oobAlpha: 0.0 // transparent outside
            )
            enc.setFragmentBytes(&U, length: MemoryLayout<WarpUniforms>.stride, index: 0)

            // Fullscreen triangle: 3 vertices, no buffers
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        
        func applyHomographyAccelerate(to ciImage: CIImage, H: simd_float3x3) -> CIImage? {
            let start = Date()
            let context = CIContext(options: nil)
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                return nil
            }
            do {
                
                var format = vImage_CGImageFormat(
                    bitsPerComponent: 8,
                    bitsPerPixel: 8 * 4,
                    colorSpace: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue))!
                // TODO: we need to be smarter about how we calculate this scale factor (depends a bit on the size of the bounding box on the screen)
                // TODO: compute the bounding box of the scrollView projected onto the image
                let backgroundImage = makeBlackImage(matching: cgImage)!
                let backgroundBuffer = try vImage.PixelBuffer<vImage.Interleaved8x4>(
                    cgImage: backgroundImage,
                    cgImageFormat: &format)
                let foregroundBuffer = try vImage.PixelBuffer<vImage.Interleaved8x4>(
                    cgImage: cgImage,
                    cgImageFormat: &format)
                let warpedBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(
                    size: backgroundBuffer.size)

                let dstPoints: [vImagePoint] = {
                    func map(_ p: CGPoint, subtracting: vImagePoint) -> vImagePoint {
                        // scale
                        let v = SIMD3(Float(p.x) * Float(cgImage.width), Float(p.y) * Float(cgImage.height), 1)
                        let w = H.inverse * v
                        return (w.x / w.z - subtracting.0, Float(backgroundImage.height) - w.y / w.z - subtracting.1)
                    }
                    let dstCenter = map(CGPoint(x: 0.5, y: 0.5), subtracting: (0.0, 0.0))
                    let centerOffset = (dstCenter.0 - Float(backgroundImage.width)/2.0,
                                        dstCenter.1 - Float(backgroundImage.height)/2.0)
                    var dstTopLeft = map(CGPoint(x: 0.0,y: 0.0), subtracting: centerOffset)
                    var dstTopRight = map(CGPoint(x: 1.0,y: 0.0), subtracting: centerOffset)
                    var dstBottomLeft = map(CGPoint(x: 0.0,y: 1.0), subtracting: centerOffset)
                    var dstBottomRight = map(CGPoint(x: 1.0,y: 1.0), subtracting: centerOffset)
                    let centerFinal = map(CGPoint(x: 0.5, y: 0.5), subtracting: centerOffset)
                    let a = calcArea(vertices: [dstTopLeft, dstBottomLeft, dstBottomRight, dstTopRight])
                    let sourceArea = Float(cgImage.width * cgImage.height)
                    let scaleFactor = sqrt(sourceArea / a)
                    // use scaleFactor to scale about the center point
                    dstTopLeft = scale(dstTopLeft, around: centerFinal, byFactor: scaleFactor)
                    dstTopRight = scale(dstTopRight, around: centerFinal, byFactor: scaleFactor)
                    dstBottomLeft = scale(dstBottomLeft, around: centerFinal, byFactor: scaleFactor)
                    dstBottomRight = scale(dstBottomRight, around: centerFinal, byFactor: scaleFactor)

                    print("area", a)
                    print(dstTopLeft, dstBottomLeft, dstBottomRight, dstTopRight)
                    
                    return [dstTopLeft, dstTopRight, dstBottomLeft, dstBottomRight]
                }()
                
                let srcPoints: [vImagePoint] = {
                    let foregroundWidth = Float(cgImage.width)
                    let foregroundHeight = Float(cgImage.height)
                    
                    let srcTopLeft: (Float, Float) = (0, foregroundHeight)
                    let srcTopRight: (Float, Float) = (foregroundWidth, foregroundHeight)
                    let srcBottomLeft: (Float, Float) = (0, 0)
                    let srcBottomRight: (Float, Float) = (foregroundWidth, 0)
                    print(srcTopLeft, srcBottomLeft, srcBottomRight, srcTopRight)
                    return [srcTopLeft, srcTopRight, srcBottomLeft, srcBottomRight]
                }()
                
                var transform = vImage_PerpsectiveTransform()
                vImageGetPerspectiveWarp(srcPoints, dstPoints, &transform, 0)
                foregroundBuffer.withUnsafePointerToVImageBuffer { src in
                    warpedBuffer.withUnsafePointerToVImageBuffer { dst in
                        
                        var bgColor: [UInt8] = [0, 0, 0, 0]
                        
                        vImagePerspectiveWarp_ARGB8888(
                            src, dst, nil,
                            &transform,
                            vImage_WarpInterpolation(kvImageInterpolationLinear),
                            &bgColor,
                            vImage_Flags(kvImageBackgroundColorFill))
                    }
                }
                backgroundBuffer.alphaComposite(.nonpremultiplied,
                                                topLayer: warpedBuffer,
                                                destination: backgroundBuffer)
                
                
                let result = backgroundBuffer.makeCGImage(cgImageFormat: format)
                print(Date().timeIntervalSince(start))
                return CIImage(cgImage: result!)
            } catch {
                return nil
            }
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
        
        func applyHomographyToImage(ci: CIImage, H: simd_float3x3)->CIImage? {
            if renderer == nil {
                renderer = WarpRenderer()
            }
            guard let renderer = renderer, let overlay = overlay else {
                return nil
            }
            return renderer.processFrame(ciImage: ci,
                                  H: H.inverse)
            return nil
            return applyHomographyAccelerate(to: ci, H: H)
            // Map via H (Core Image coords: origin at bottom-left)
            func map(_ p: CGPoint) -> CGPoint {
                let v = SIMD3(Float(p.x), Float(p.y), 1)
                let w = H * v
                return CGPoint(x: CGFloat(w.x / w.z), y: CGFloat(w.y / w.z))
            }
            let r = ci.extent
            // might want to oversample
            let targetRect = r
            let bl = map(CGPoint(x: r.minX, y: r.minY))
            let br = map(CGPoint(x: r.maxX, y: r.minY))
            let center = map(CGPoint(x: (r.maxX + r.minY)/2.0, y: (r.maxY + r.minY)/2.0))
            let tl = map(CGPoint(x: r.minX, y: r.maxY))
            let tr = map(CGPoint(x: r.maxX, y: r.maxY))

            // 1) Compute the bounding box of the warped quad
            let xs = [tl.x, tr.x, br.x, bl.x]
            let ys = [tl.y, tr.y, br.y, bl.y]
            let minX = xs.min()!, maxX = xs.max()!
            let minY = ys.min()!, maxY = ys.max()!
            let bbox = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
            let (sx, sy, padX, padY): (CGFloat, CGFloat, CGFloat, CGFloat)

            // make smaller for testing
            let sxFill = targetRect.width  / bbox.width
            let syFill = targetRect.height / bbox.height
            
            // Uniform scale (preserve aspect). Center with padding.
            let s = min(sxFill, syFill)
            print("s \(s)")
            sx = s; sy = s

            
            // this preserves the placement of the center at the center point of the new bounding box
            let tx = -center.x + 1.0 / s * targetRect.width / 2.0
            let ty = -center.y + 1.0 / s * targetRect.height / 2.0
            print(r.minX, r.minY)
            // Apply: translate to origin -> scale -> translate into targetRect
            func norm(_ p: CGPoint) -> CGPoint {
                let x = (p.x + tx) * sx + targetRect.minX
                let y = (p.y + ty) * sy + targetRect.minY
                return CGPoint(x: x, y: y)
            }

            let TL = norm(tl)
            let TR = norm(tr)
            let BR = norm(br)
            let BL = norm(bl)
            print(norm(center))
            

            // 3) Use CIPerspectiveTransformWithExtent so the OUTPUT extent is fixed
            guard let f = CIFilter(name: "CIPerspectiveTransform") else {
                return nil
            }
            f.setValue(ci, forKey: kCIInputImageKey)
            f.setValue(CIVector(cgPoint: TL), forKey: "inputTopLeft")
            f.setValue(CIVector(cgPoint: TR), forKey: "inputTopRight")
            f.setValue(CIVector(cgPoint: BR), forKey: "inputBottomRight")
            f.setValue(CIVector(cgPoint: BL), forKey: "inputBottomLeft")
            // Crop to original dimensions
            let quadBoundingBox = CGRect(
                x: r.minX,
                y: r.minY,
                width: r.width,
                height: r.height
            )
            return f.outputImage?.cropped(to: quadBoundingBox)
        }
        
        /// Adjust the UIImage based on the gravity vector as returned by the CoreMotion.
        /// We assume a landscape right orientation for the image
        /// - Parameters:
        ///   - uiImage: the UIImage (e.g., from the camera feed or a freeze frame)
        /// - Returns: the corrected CGImage if the transformation can be applied successfully (nil otherwise)
        func rectifyToLevel(_ cg: CGImage) -> CGImage? {
            guard doPerspectiveCorrection else {
                // short circuiting and just returning input
                return cg
            }
            guard let gravity = gravity else {
                return nil
            }
            let ci = CIImage(cgImage: cg)
            guard let K = self.K else {
                // If intrinsics missing, just return input
                return nil
            }
            // TODO: maybe we can negate each of these entries and negate the from: axis
            let gravityVectorInCameraConventions = simd_float3(Float(gravity.y), Float(-gravity.x), Float(-gravity.z))
            let R = simd_float3x3(simd_quatf(from: simd_float3(0, 0, 1), to: gravityVectorInCameraConventions))
            var K_scaled = K
            let s = Float(Self.destinationTextureScaleFactor)
            K_scaled.columns.0.x *= s
            K_scaled.columns.1.y *= s
            K_scaled.columns.2.x *= s
            K_scaled.columns.2.y *= s
            print("K_scaled", K_scaled)
            let H = K * R * simd_inverse(K_scaled)
            // correct for perspective
            guard let out = applyHomographyToImage(ci: ci, H: H.inverse) else {
                return nil
            }
            // the output as a CGImage
            guard let outCG = ciCtx.createCGImage(out, from: out.extent) else {
                return nil
            }
            return outCG
        }

        init(session: AVCaptureSession) {
            self.session = session
            self.library = try! device.makeDefaultLibrary()!
            psDesc.vertexFunction = library.makeFunction(name: "vs_fullscreen_triangle")
            psDesc.fragmentFunction = library.makeFunction(name: "fs_warp")
            psDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            self.pipeline = try! device.makeRenderPipelineState(descriptor: psDesc)
            self.sampler = device.makeSamplerState(descriptor: MTLSamplerDescriptor())! // default = linear, clamp
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
            out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
            // fast
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
        
        // Convert frames to UIImage and keep only the latest
        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
            // We are careful not to process frames too slowly, so we discard frames if we are still processing a previous frame
            guard !processingFrame else {
                return
            }
            processingFrame = true
            DispatchQueue.global(qos: .userInteractive).async { [self] in
                guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                    self.processingFrame = false
                    return
                }
                self.updateIntrinsicsIfNeeded(from: pb)
                let ci = CIImage(cvPixelBuffer: pb)
                // Create CGImage and orient it to match the preview
                guard let cg = self.ciContext.createCGImage(ci, from: ci.extent) else {
                    self.processingFrame = false
                    return
                }
                
                let oriented: UIImage.Orientation = .right
                // landscapeRight
                let image = UIImage(cgImage: cg, scale: 1, orientation: oriented)
                self.latestImage = image
                if let filteredImage = self.applyFiltering(image.cgImage!)  {
                    let startLevel = Date()
                    if let leveled = self.rectifyToLevel(filteredImage) {
                        print("time to level \(Date().timeIntervalSince(startLevel))")
                        self.preview?.display(cgImage:leveled)
                    } else {
                        self.preview?.display(cgImage:filteredImage)
                    }
                }

                self.processingFrame = false
            }
        }
        
        func setFilterMode(_ filterMode: FilterMode) {
            self.filterMode = filterMode
        }
        
        private func colorizeBinary(binary: Mat, fgRGB: Scalar , bgRGB: Scalar )->Mat  {
            let out  = Mat(size: binary.size(), type: CvType.CV_8UC3);
            out.setTo(scalar: bgRGB)
            out.setTo(scalar: fgRGB, mask: binary)
            return out
        }
        
        func applyFiltering(_ input: CGImage)->CGImage? {
            guard filterMode != .none else {
                return input
            }
            // Show source image
            let src = Mat(cgImage: input)

            // Transform source image to gray if it is not already
            let gray: Mat
            let thresholded: Mat

            if (src.channels() >= 3) {
                gray = Mat()
                Imgproc.cvtColor(src: src, dst: gray, code: .COLOR_BGR2GRAY)
                
                thresholded = Mat()

                Imgproc.adaptiveThreshold(src: gray, dst: thresholded, maxValue: 255, adaptiveMethod: .ADAPTIVE_THRESH_MEAN_C, thresholdType: .THRESH_BINARY, blockSize: 161, C: 30)

            } else {
                thresholded = src
            }
            let fgRGB: Scalar
            let bgRGB: Scalar

            switch filterMode {
            case .blackOnWhite:
                fgRGB = Scalar(255.0, 255.0, 255.0)
                bgRGB = Scalar(0.0, 0.0, 0.0)
            case .whiteOnBlack:
                fgRGB = Scalar(0.0, 0.0, 0.0)
                bgRGB = Scalar(255.0, 255.0, 255.0)
            case .yellowOnBlack:
                fgRGB = Scalar(0.0, 0.0, 0.0)
                bgRGB = Scalar(255.0, 255.0, 0.0)
            case .none:
                // Note: this is impossible to get to
                fgRGB = Scalar(255.0, 255.0, 255.0)
                bgRGB = Scalar(0.0, 0.0, 0.0)
            }
            
            let coloredMat = colorizeBinary(binary: thresholded, fgRGB: fgRGB, bgRGB: bgRGB)
            return coloredMat.toCGImage()
        }
        
        // Toggle frozen state
        func applyFreeze(_ frozen: Bool) {
            guard let overlay, let preview else {
                return
            }
            if frozen {
                let imageToRotate: UIImage
                if let img = latestImage {
                    imageToRotate = img
                } else {
                    imageToRotate = fallbackSnapshot(of: preview)!
                }
                if let filteredImage = self.applyFiltering(imageToRotate.cgImage!), let leveled = rectifyToLevel(filteredImage) {
                    let leveledUIImage = UIImage(cgImage: leveled)
                    overlay.image = leveledUIImage
                    overlay.isHidden = false
                    preview.isHidden = true
                }
            } else {
                overlay.isHidden = true
                preview.isHidden = false
            }
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
        // Fallback snapshot if no video frame was captured yet
        private func fallbackSnapshot(of view: UIView) -> UIImage? { // Note: AV layers sometimes render black via snapshot; this is just a fallback.
            let renderer = UIGraphicsImageRenderer(bounds: view.bounds)
            return renderer.image { ctx in
                view.drawHierarchy(in: view.bounds, afterScreenUpdates: false)
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

// MARK: - Demo
struct RearWideCameraDemo_Previews: PreviewProvider {
    static var previews: some View {
        RearWideCameraView()
    }
}
