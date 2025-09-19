//
// RearWideCameraView.swift
// case4visionmagnifier
//
// Created by Paul Ruvolo on 8/13/25.

import SwiftUI
import AVFoundation
import UIKit
import Combine
import simd
import CoreMotion
import CoreImage
// MARK: - SwiftUI Camera View
struct RearWideCameraView: View {
    @StateObject private var model = CameraModel()
    @State private var isFrozen = false
    @State private var includeGuides = false
    var body: some View {
        ZStack {
            if let session = model.session {
                CameraPreview(session: session, isFrozen: $isFrozen)
                    .ignoresSafeArea()
                    .onAppear { model.start() }
                    .onDisappear { model.stop() }
                VStack {
                    HStack {
                        Button(isFrozen ? "Unfreeze" : "Freeze") {
                            isFrozen.toggle()
                        }.buttonStyle(.borderedProminent)
                        Spacer()
                        Button("Color Filter") {
                            print("Top Right tapped")
                        }.buttonStyle(.borderedProminent)
                    }
                    Spacer()
                    HStack {
                        Button("Toggle Flash Light") {
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
                                    print("torch error")
                                }
                            } else if device.torchMode == .on {
                                do {
                                    try device.lockForConfiguration()
                                    device.torchMode = .off
                                    device.unlockForConfiguration()
                                } catch {
                                    device.unlockForConfiguration()
                                    print("torch error")
                                }
                            }
                        }.buttonStyle(.borderedProminent)
                        Spacer()
                        Button("Toggle Guide Lines") {
                            includeGuides.toggle()
                        }.buttonStyle(.borderedProminent)
                    }
                }
                .padding() // space from edges
               
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
                        .position(x: geo.size.width/2, y: 100)
                    Rectangle()
                        .fill(Color.red)
                        .frame(width: geo.size.width, height: 10)
                        .position(x: geo.size.width/2, y: geo.size.height-100)
                }.ignoresSafeArea()
            }
        }
        .alert(item: $model.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("OK")))
        }.onAppear {
            AppDelegate.orientationLock = .landscape
            // Nudge rotation if already in portrait:
            UIDevice.current.setValue(UIInterfaceOrientation.landscapeRight.rawValue, forKey: "orientation")
            UINavigationController.attemptRotationToDeviceOrientation()
        }.onDisappear {
            AppDelegate.orientationLock = .all // restore
        }
    }
}

//final class PreviewView: UIView {
//    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
//    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer
//    }
//}

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
    var minZoom: CGFloat = 1.0
    var maxZoom: CGFloat = 6.0
    
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
    
    func updateUIView(_ scroll: UIScrollView, context: Context) { context.coordinator.applyFreeze(isFrozen)
        context.coordinator.centerContent()
    }
    
    // MARK: - Coordinator
    
    final class Coordinator: NSObject, UIScrollViewDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
        private weak var session: AVCaptureSession?
        weak var scrollView: UIScrollView?
        weak var container: UIView?
        weak var preview: PreviewView?
        weak var overlay: UIImageView?
        private let ciContext = CIContext()
        private let sampleQueue = DispatchQueue(label: "freeze.preview.sample")
        private var latestImage: UIImage?
        private let motion = CMMotionManager()
        private(set) var gravity: CMAcceleration?
        // camera intrinsics (pixels)
        private var K: simd_float3x3?
        private var processingFrame = false
        private let ciCtx = CIContext(options: nil)
        
        
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
            // Map via H (Core Image coords: origin at bottom-left)
            func map(_ p: CGPoint) -> CGPoint {
                let v = SIMD3(Float(p.x), Float(p.y), 1)
                let w = H * v
                return CGPoint(x: CGFloat(w.x / w.z), y: CGFloat(w.y / w.z))
            }
            let r = ci.extent
            let targetRect = r
            let bl = map(CGPoint(x: r.minX, y: r.minY))
            let br = map(CGPoint(x: r.maxX, y: r.minY))
            let tl = map(CGPoint(x: r.minX, y: r.maxY))
            let tr = map(CGPoint(x: r.maxX, y: r.maxY))

            // 1) Compute the bounding box of the warped quad
            let xs = [tl.x, tr.x, br.x, bl.x]
            let ys = [tl.y, tr.y, br.y, bl.y]
            let minX = xs.min()!, maxX = xs.max()!
            let minY = ys.min()!, maxY = ys.max()!
            let bbox = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)

            // 2) Build a normalization that places the quad inside targetRect
            let tx = -bbox.minX
            let ty = -bbox.minY
            let sxFill = targetRect.width  / bbox.width
            let syFill = targetRect.height / bbox.height

            let (sx, sy, padX, padY): (CGFloat, CGFloat, CGFloat, CGFloat)
            // Uniform scale (preserve aspect). Center with padding.
            let s = min(sxFill, syFill)
            sx = s; sy = s
            padX = (targetRect.width  - bbox.width  * s) * 0.5
            padY = (targetRect.height - bbox.height * s) * 0.5
            // Apply: translate to origin -> scale -> translate into targetRect
            func norm(_ p: CGPoint) -> CGPoint {
                let x = (p.x + tx) * sx + targetRect.minX + padX
                let y = (p.y + ty) * sy + targetRect.minY + padY
                return CGPoint(x: x, y: y)
            }

            let TL = norm(tl)
            let TR = norm(tr)
            let BR = norm(br)
            let BL = norm(bl)

            // 3) Use CIPerspectiveTransformWithExtent so the OUTPUT extent is fixed
            guard let f = CIFilter(name: "CIPerspectiveTransformWithExtent") else {
                return nil
            }
            f.setValue(ci, forKey: kCIInputImageKey)
            f.setValue(CIVector(cgRect: targetRect), forKey: "inputExtent")
            f.setValue(CIVector(cgPoint: TL), forKey: "inputTopLeft")
            f.setValue(CIVector(cgPoint: TR), forKey: "inputTopRight")
            f.setValue(CIVector(cgPoint: BR), forKey: "inputBottomRight")
            f.setValue(CIVector(cgPoint: BL), forKey: "inputBottomLeft")
            return f.outputImage
        }
        
        /// Adjust the UIImage based on the gravity vector as returned by the CoreMotion.
        /// We assume a landscape right orientation for the image
        /// - Parameters:
        ///   - uiImage: the UIImage (e.g., from the camera feed or a freeze frame)
        /// - Returns: the corrected CGImage if the transformation can be applied successfully (nil otherwise)
        func rectifyToLevel(_ uiImage: UIImage) -> CGImage? {
            guard let cg = uiImage.cgImage, let gravity = gravity else {
                return nil
            }
            let ci = CIImage(cgImage: cg)
            guard let K = self.K else {
                // If intrinsics missing, just return input
                return nil
            }
            let gravityVectorInCameraConventions = simd_float3(Float(gravity.y), Float(-gravity.x), Float(-gravity.z))
            let R = simd_float3x3(simd_quatf(from: simd_float3(0, 0, 1), to: gravityVectorInCameraConventions))
            let H = K * R * simd_inverse(K)
            // correct for perspective
            guard let out = applyHomographyToImage(ci: ci, H: H) else {
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
            DispatchQueue.global(qos: .userInteractive).async {
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
                // Ensure .up orientation
                let upright = UIImage(cgImage: image.cgImage!, scale: image.scale, orientation: .up)
                if let leveled = self.rectifyToLevel(upright)  {
                    self.preview?.display(cgImage:leveled)
                }
                self.processingFrame = false
            }
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
                // Ensure .up orientation
                let upright = UIImage(cgImage: imageToRotate.cgImage!,
                                      scale: imageToRotate.scale,
                                      orientation: .up)
                if let leveled = rectifyToLevel(upright) {
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
                    let session = try Self.makeRearWideHighResSession()
                    DispatchQueue.main.async {
                        self.session = session
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
    
    private static func makeRearWideHighResSession() throws -> AVCaptureSession {
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
        device.unlockForConfiguration()
        // Add a dummy output so the session can run even if you later want to add data/video outputs.
        // For preview-only, this isn’t required, but harmless.
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        return session
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
