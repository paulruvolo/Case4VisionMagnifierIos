//
//  RearWideCameraView.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 8/13/25.
//


import SwiftUI
import AVFoundation
import UIKit
import Combine

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
                Button(isFrozen ? "Unfreeze" : "Freeze") { isFrozen.toggle() }
                    .buttonStyle(.borderedProminent)
                    .padding()
            } else {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Preparing camera…")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .task { await model.configure() }
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
                }
                .ignoresSafeArea()
            }
        }
        .alert(item: $model.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("OK")))
        }
        .onAppear {
            AppDelegate.orientationLock = .landscape
            // Nudge rotation if already in portrait:
            UIDevice.current.setValue(UIInterfaceOrientation.landscapeRight.rawValue,
                                      forKey: "orientation")
            UINavigationController.attemptRotationToDeviceOrientation()
        }
        .onDisappear {
            AppDelegate.orientationLock = .all // restore
        }
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}


struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    @Binding var isFrozen: Bool
    var minZoom: CGFloat = 1.0
    var maxZoom: CGFloat = 6.0

    func makeCoordinator() -> Coordinator { Coordinator(session: session) }

    func makeUIView(context: Context) -> UIScrollView {
        let scroll = UIScrollView()
        scroll.minimumZoomScale = minZoom
        scroll.maximumZoomScale = maxZoom
        scroll.bouncesZoom = true
        scroll.showsVerticalScrollIndicator = false
        scroll.showsHorizontalScrollIndicator = false
        scroll.backgroundColor = .black
        scroll.delegate = context.coordinator
        
        scroll.isScrollEnabled = false

        // Container so both preview and overlay zoom together
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
        
        // Initial recenter after the first layout pass
        DispatchQueue.main.async {
            context.coordinator.centerContent()
            context.coordinator.lockContentOffsetToCenter()
        }

        return scroll
    }

    func updateUIView(_ scroll: UIScrollView, context: Context) {
        context.coordinator.applyFreeze(isFrozen)
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

        init(session: AVCaptureSession) {
            self.session = session
        }

        // Add (or reuse) a VideoDataOutput so we can grab the latest frame as UIImage
        func ensureVideoOutputOnSession() {
            guard let session = session else { return }

            // Reuse if one exists
            if session.outputs.contains(where: { $0 is AVCaptureVideoDataOutput }) {
                if let out = session.outputs.compactMap({ $0 as? AVCaptureVideoDataOutput }).first {
                    out.setSampleBufferDelegate(self, queue: sampleQueue)
                }
                return
            }

            let out = AVCaptureVideoDataOutput()
            out.alwaysDiscardsLateVideoFrames = true
            out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange] // fast
            if session.canAddOutput(out) {
                session.addOutput(out)
            }
            out.setSampleBufferDelegate(self, queue: sampleQueue)

            // Match orientation to preview
            if let conn = out.connection(with: .video) {
                if #available(iOS 17.0, *) {
                    if conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }
                } else if conn.isVideoOrientationSupported {
                    conn.videoOrientation = .landscapeRight
                }
            }
        }

        // Convert frames to UIImage and keep only the latest
        func captureOutput(_ output: AVCaptureOutput,
                           didOutput sampleBuffer: CMSampleBuffer,
                           from connection: AVCaptureConnection) {
            guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let ci = CIImage(cvPixelBuffer: pb)

            // Create CGImage and orient it to match the preview
            guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return }

            let oriented: UIImage.Orientation = .right // landscapeRight
            let image = UIImage(cgImage: cg, scale: 1, orientation: oriented)
            latestImage = image
        }

        // Toggle frozen state
        func applyFreeze(_ frozen: Bool) {
            guard let overlay, let preview else { return }

            if frozen {
                // Prefer camera frame; if not available, fall back to a view snapshot
                let imageToRotate: UIImage
                if let img = latestImage {
                    imageToRotate = img
                } else {
                    imageToRotate = fallbackSnapshot(of: preview)!
                }
                
                let rotated = UIImage(cgImage: imageToRotate.cgImage!, scale: imageToRotate.scale, orientation: .up)
                overlay.image = rotated
                overlay.isHidden = false
                preview.isHidden = true
            } else {
                overlay.isHidden = true
                preview.isHidden = false
            }
            // ⬇️ Recenter after the visibility change, even if no zoom yet.
            DispatchQueue.main.async { [weak self] in
                guard let self = self, let scroll = self.scrollView else { return }
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
        private func fallbackSnapshot(of view: UIView) -> UIImage? {
            // Note: AV layers sometimes render black via snapshot; this is just a fallback.
            let renderer = UIGraphicsImageRenderer(bounds: view.bounds)
            return renderer.image { ctx in
                view.drawHierarchy(in: view.bounds, afterScreenUpdates: false)
            }
        }

        // MARK: UIScrollViewDelegate

        func viewForZooming(in scrollView: UIScrollView) -> UIView? { container }

        func scrollViewDidZoom(_ scrollView: UIScrollView) {
            centerContent()
            lockContentOffsetToCenter()
        }
        
        // Center the zoomed content using insets (works when content is smaller than bounds)
        func centerContent() {
            guard let scroll = scrollView else { return }
            // If you rely on contentSize == container.bounds.size this stays correct.
            let contentSize = scroll.contentSize
            let dx = max(0, (scroll.bounds.width  - contentSize.width)  / 2)
            let dy = max(0, (scroll.bounds.height - contentSize.height) / 2)
            scroll.contentInset = UIEdgeInsets(top: dy, left: dx, bottom: dy, right: dx)
        }

        // Keep the visual center fixed; prevents any drift while zooming
        func lockContentOffsetToCenter() {
            guard let scroll = scrollView else { return }
            let contentSize = scroll.contentSize
            let target = CGPoint(
                x: max(0, (contentSize.width  - scroll.bounds.width ) / 2) - scroll.adjustedContentInset.left,
                y: max(0, (contentSize.height - scroll.bounds.height) / 2) - scroll.adjustedContentInset.top
            )
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
            alert = .init(title: "Camera Access Denied",
                          message: "Enable camera access in Settings → Privacy → Camera.")
            return
        }

        // Build the session off the main thread
        await withCheckedContinuation { continuation in
            sessionQueue.async {
                do {
                    let session = try Self.makeRearWideHighResSession()
                    DispatchQueue.main.async {
                        self.session = session
                        continuation.resume()
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.alert = .init(title: "Camera Error", message: error.localizedDescription)
                        continuation.resume()
                    }
                }
            }
        }
    }

    func start() {
        guard let session = session, !session.isRunning else { return }
        sessionQueue.async { session.startRunning() }
    }

    func stop() {
        guard let session = session, session.isRunning else { return }
        sessionQueue.async { session.stopRunning() }
    }

    // MARK: Session builder (rear WIDE, highest resolution)
    private static func makeRearWideHighResSession() throws -> AVCaptureSession {
        let session = AVCaptureSession()
        // Use inputPriority so our chosen activeFormat takes precedence over preset.
        session.sessionPreset = .inputPriority

        // Pick the BACK WIDE-ANGLE camera (NOT ultra wide).
        // Apple device types:
        // - .builtInWideAngleCamera   → the standard "Wide" camera (≈ 26mm equiv)
        // - .builtInUltraWideCamera   → the Ultra Wide (≈ 13mm) — we intentionally avoid this
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .back
        )

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
            if areaA != areaB { return areaA > areaB }
            // Tie-breaker: higher max FPS
            let maxA = a.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            let maxB = b.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            return maxA > maxB
        }

        guard let top = sorted.first else { return (device.activeFormat, nil) }
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
