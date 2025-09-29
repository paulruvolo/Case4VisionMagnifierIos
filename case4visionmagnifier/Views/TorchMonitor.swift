import SwiftUI
import AVFoundation

final class TorchMonitor: NSObject, ObservableObject {
    @Published private(set) var isOn = false
    @Published private(set) var isAvailable = false

    private var device: AVCaptureDevice?
    private var kvo: NSKeyValueObservation?

    /// Call this once you have a running session
    func attach(device: AVCaptureDevice) {
        // Pick the device you actually use in your pipeline
        bind(to: device)
    }

    /// If your app can switch cameras, call this again with the new device.
    private func bind(to device: AVCaptureDevice) {
        // tear down previous observation
        kvo?.invalidate()

        self.device = device
        self.isAvailable = device.hasTorch
        self.isOn = device.isTorchActive

        // Observe real torch state (KVO)
        kvo = device.observe(\.isTorchActive, options: [.initial, .new]) { [weak self] dev, _ in
            DispatchQueue.main.async { self?.isOn = dev.isTorchActive }
        }
    }

    func toggleTorch(level: Float = 1.0) {
        guard let device, device.hasTorch else { return }
        do {
            try device.lockForConfiguration()
            if device.isTorchActive {
                device.torchMode = .off
            } else {
                try device.setTorchModeOn(level: min(max(level, 0.0), 1.0))
            }
            device.unlockForConfiguration()
        } catch {
            // handle/log
        }
    }

    deinit { kvo?.invalidate() }
}
