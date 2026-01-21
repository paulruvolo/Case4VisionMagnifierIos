//
//  case4visionmagnifierApp.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 8/13/25.
//

import SwiftUI
import UIKit


// 1) App delegate exposes a mutable orientation lock
class AppDelegate: NSObject, UIApplicationDelegate {
    static var orientationLock = UIInterfaceOrientationMask.landscapeRight

    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?)
        -> UIInterfaceOrientationMask
    {
        Self.orientationLock
    }
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        return true
    }
}

@main
struct case4visionmagnifierApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @State var activating = false
    @State var code: String?
    @State var promptToScanCodeOrBuyProduct = false
    @ObservedObject var iapManager = IAPManager.shared
    
    init() {
        // Ensure trial is started at first launch / whatever trigger you want
        TrialStore.startTrialIfNeeded()
    }
    var body: some Scene {
        // only for testing
        let trialExpired = TrialStore.isTrialExpired(days: 0)
        WindowGroup {
            if !iapManager.isPurchased && (promptToScanCodeOrBuyProduct || trialExpired) {
                ScrollView {
                    VStack(spacing: 16) {
                        Text("To Continue Using the App, Either Scan Your QR Code that Came With CaseForVision or Purchase Full Usage of the App Below")
                            .font(.largeTitle)
                            .bold()
                            .multilineTextAlignment(.center)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity)

                        PurchaseView()
                    }
                    .padding()
                }
            } else {
                RearWideCameraView()
            }
        }
    }
}
