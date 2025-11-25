//
//  case4visionmagnifierApp.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 8/13/25.
//

import SwiftUI
import UIKit
import FirebaseCore
import FirebaseAuth


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
        FirebaseApp.configure()
        AuthHandler.shared.createAuthHandle()
        return true
    }
    
    func application(_ application: UIApplication,
                     didReceiveRemoteNotification userInfo: [AnyHashable : Any],
                     fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {

        if Auth.auth().canHandleNotification(userInfo) {
            completionHandler(.noData)
            return
        }

        completionHandler(.noData)
    }
    
    func application(_ application: UIApplication,
                     didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {

        #if DEBUG
        Auth.auth().setAPNSToken(deviceToken, type: .sandbox)
        #else
        Auth.auth().setAPNSToken(deviceToken, type: .prod)
        #endif
    }
}

@main
struct case4visionmagnifierApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @ObservedObject var authHandler = AuthHandler.shared
    @State var activating = false
    @State var code: String?
    @State var promptToScanCodeOrBuyProduct = false
    
    init() {
        // Ensure trial is started at first launch / whatever trigger you want
        TrialStore.startTrialIfNeeded()
    }
    var body: some Scene {
        let trialExpired = TrialStore.isTrialExpired(days: 7)

        WindowGroup {
            Group {
                if activating {
                    if authHandler.currentUID == nil {
                        PhoneLoginView()
                    }
                } else if promptToScanCodeOrBuyProduct || (trialExpired || AuthHandler.disableUsageDuringTrial) && !authHandler.isActivated {
                    Text("To Continue Using the App, Either Scan Your Activation QR Code that Came With Case4Vision or Buy the Full Version of the App")
                        .font(.largeTitle)
                        .bold()
                } else {
                    RearWideCameraView()
                }
            }
            .onContinueUserActivity(NSUserActivityTypeBrowsingWeb) { activity in
                 guard let url = activity.webpageURL else { return }
                 let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
                 if let code = components?.queryItems?.first(where: { $0.name == "code" })?.value {
                     guard !authHandler.isActivated else {
                         // already activated
                         return
                     }
                     self.code = code
                     if authHandler.currentUID != nil {
                         Task {
                             if await authHandler.associateCodeWithAccount(code: code) {
                                 authHandler.recordActivationInKeyChain()
                                 activating = false
                             } else {
                                 // TODO: propagate some sort of error
                             }
                         }
                     } else {
                         promptToScanCodeOrBuyProduct = false
                         activating = true
                     }
                 }
             }
             .onChange(of: authHandler.currentUID) { (_, newValue) in
                 guard let code else {
                     promptToScanCodeOrBuyProduct = true
                     return
                 }
                 
                 Task {
                     if await authHandler.associateCodeWithAccount(code: code) {
                         authHandler.recordActivationInKeyChain()
                         activating = false
                     } else {
                         // TODO: propagate some sort of error
                     }
                 }
             }
        }
    }
}
