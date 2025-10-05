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
}

@main
struct case4visionmagnifierApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            RearWideCameraView()
        }
    }
}
