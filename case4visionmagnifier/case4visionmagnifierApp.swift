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
    
    var body: some Scene {
        WindowGroup {
            Group {
                if !activating {
                    RearWideCameraView()
                } else {
                    if authHandler.currentUID == nil {
                        PhoneLoginView()
                    }
                }
            }
            // This is for universal links
             .onContinueUserActivity(NSUserActivityTypeBrowsingWeb) { activity in
                 guard let url = activity.webpageURL else { return }
                 let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
                 if let code = components?.queryItems?.first(where: { $0.name == "code" })?.value {
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
                         activating = true
                     }
                 }
             }
             .onChange(of: authHandler.currentUID) { (_, newValue) in
                 guard let code else { return }
                 
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
