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
    static let trialLimit = 14
    @State var showAlert = false
    @AppStorage("lastWarned") var lastWarned: Int?
    
    init() {
        // Ensure trial is started at first launch / whatever trigger you want
        TrialStore.startTrialIfNeeded()
    }
    var body: some Scene {
        let trialExpired = TrialStore.isTrialExpired()
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
                RearWideCameraView().onAppear {
                    guard !iapManager.isPurchased else {
                        return
                    }
                    guard let lastWarned else {
                        warnAboutTrial()
                        return
                    }
                    let trialRemaining = TrialStore.trialDaysLeft()
                    if trialRemaining < lastWarned {
                        warnAboutTrial()
                    }
                }.alert("You are using a trial version of the Q++", isPresented: $showAlert) {
                    Button("OK", role: .cancel) {
                        
                    }
                } message: {
                    Text("Your trial will expire in \(TrialStore.trialDaysLeft()) days. If you bought Case for Vision, scan your code to activate the full version.")
                }
            }
        }

    }
    func warnAboutTrial() {
        DispatchQueue.main.async {
            self.lastWarned = TrialStore.trialDaysLeft()
            showAlert = true
        }
    }
}
