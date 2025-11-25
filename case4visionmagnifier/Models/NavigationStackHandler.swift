//
//  NavigationStackHandler.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 11/21/25.
//


import Foundation
import SwiftUI


enum NavigationDestinations: Hashable {
    case enterPhoneNumber
    case verifyPhoneNumber
    
    @ViewBuilder
    var asView: some View {
        switch self {
        case .enterPhoneNumber:
            EnterPhoneNumberView()
        case .verifyPhoneNumber:
            VerifyPhoneNumberView()
        }
    }
}

/// This is a class that can be used for managing the navigation stack
class NavigationStackHandler: ObservableObject {
    @Published var path: [NavigationDestinations] {
        didSet {
            print("navstack \(path.count)")
        }
    }
    
    init() {
        path = []
    }

    func pushView(_ newValue: NavigationDestinations) {
        path.append(newValue)
    }
    
    func popToHome() {
        path.removeLast(path.count)
    }
    
    func pop() {
        path.removeLast()
    }
}
