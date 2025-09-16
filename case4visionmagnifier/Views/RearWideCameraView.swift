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
    @State private var includeGuides = true
    var body: some View {
        ZStack {
            if let session = model.session {
                CameraPreview(session: session, isFrozen: $isFrozen)
                    .ignoresSafeArea()
                    .onAppear { model.start() }
                    .onDisappear { model.stop() }
                Button(isFrozen ? "Unfreeze" : "Freeze") {
                    isFrozen.toggle()
                }.buttonStyle(.borderedProminent)
                    .padding()
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

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer
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
        preview.videoPreviewLayer.session = session
        preview.videoPreviewLayer.videoGravity = .resizeAspectFill
        preview.videoPreviewLayer.connection?.videoOrientation = .landscapeRight
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
        private var K: simd_float3x3?
        // camera intrinsics (pixels)
        private let ciCtx = CIContext(options: nil)
        // ---- MOTION ----
        func startMotion() {
            guard motion.isDeviceMotionAvailable else {
                return
            }
            motion.deviceMotionUpdateInterval = 1/60
            motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] dm, _ in
                guard let self, let dm, let K = K else {
                    return
                }
                //warp.updateHomography(from: dm.gravity, K: K)
                self.gravity = dm.gravity
                print("gravity \(dm.gravity)")
            }
        }
        
        func stopMotion() {
            motion.stopDeviceMotionUpdates()
        }
        
        // ---- INTRINSICS (try real, else approximate) ----
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
        // ---- RECTIFY (gravity-leveling) ----
        func rectifyToLevel(_ uiImage: UIImage, interfaceOrientation: UIInterfaceOrientation) -> UIImage? {
            guard let cg = uiImage.cgImage, let gravity = gravity else {
                return nil
            }
            let ci = CIImage(cgImage: cg) // already oriented .up in your code
            let r = ci.extent
            guard let K = self.K else {
                return uiImage
            }
            // If intrinsics missing, just return input
            // Map device pitch/roll to view axes
            let gravityVectorInCameraConventions = simd_float3(Float(gravity.y), Float(-gravity.x), Float(-gravity.z))
            print(gravityVectorInCameraConventions)
            let R = simd_float3x3(simd_quatf(from: simd_float3(0, 0, 1), to: gravityVectorInCameraConventions))
            let H = K * R * simd_inverse(K)
            // Map the 4 corners through H (Core Image coords: origin at bottom-left)
            func map(_ p: CGPoint) -> CGPoint {
                let v = SIMD3(Float(p.x), Float(p.y), 1)
                let w = H * v
                return CGPoint(x: CGFloat(w.x / w.z), y: CGFloat(w.y / w.z))
            }
            let bl = map(CGPoint(x: r.minX, y: r.minY))
            let br = map(CGPoint(x: r.maxX, y: r.minY))
            let tl = map(CGPoint(x: r.minX, y: r.maxY))
            let tr = map(CGPoint(x: r.maxX, y: r.maxY))
            guard let f = CIFilter(name: "CIPerspectiveTransform") else {
                return nil
            }
            f.setValue(ci, forKey: kCIInputImageKey)
            f.setValue(CIVector(cgPoint: tl), forKey: "inputTopLeft")
            f.setValue(CIVector(cgPoint: tr), forKey: "inputTopRight")
            f.setValue(CIVector(cgPoint: br), forKey: "inputBottomRight")
            f.setValue(CIVector(cgPoint: bl), forKey: "inputBottomLeft")
            guard let out = f.outputImage, let outCG = ciCtx.createCGImage(out, from: out.extent) else {
                return nil
            }
            return UIImage(cgImage: outCG, scale: uiImage.scale, orientation: .up)
        }
        // ---- Helpers ----
        private func viewAlignedAngles(pitch: Float, roll: Float, io: UIInterfaceOrientation) -> (rx: Float, ry: Float) {
            switch io {
            case .landscapeRight: return (rx: pitch, ry: -roll)
            case .landscapeLeft: return (rx: -pitch, ry: roll)
            case .portrait: return (rx: roll, ry: pitch)
            case .portraitUpsideDown: return (rx: -roll, ry: -pitch)
            default:
                return (rx: roll, ry: pitch)
                // fallback
            }
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
            guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                return
            }
            updateIntrinsicsIfNeeded(from: pb)
            let ci = CIImage(cvPixelBuffer: pb)
            // Create CGImage and orient it to match the preview
            guard let cg = ciContext.createCGImage(ci, from: ci.extent) else {
                return
            }
            let oriented: UIImage.Orientation = .right
            // landscapeRight
            let image = UIImage(cgImage: cg, scale: 1, orientation: oriented)
            latestImage = image
        }
        // Toggle frozen state
        func applyFreeze(_ frozen: Bool) {
            guard let overlay, let preview else {
                return
            }
            if frozen {
                // Prefer camera frame; if not available, fall back to a view snapshot
                let imageToRotate: UIImage
                if let img = latestImage {
                    imageToRotate = img
                } else {
                    imageToRotate = fallbackSnapshot(of: preview)!
                }
                // Ensure .up orientation
                let upright = UIImage(cgImage: imageToRotate.cgImage!, scale: imageToRotate.scale, orientation: .up)
                let io = overlay.window?.windowScene?.interfaceOrientation ?? .landscapeRight
                let leveled = rectifyToLevel(upright, interfaceOrientation: io) ?? upright
                overlay.image = leveled
                overlay.isHidden = false
                preview.isHidden = true
            } else {
                overlay.isHidden = true
                preview.isHidden = false
            }
            // ⬇️ Recenter after the visibility change, even if no zoom yet.
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
        let discovery = AVCaptureDevice.DiscoverySession( deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back )
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
