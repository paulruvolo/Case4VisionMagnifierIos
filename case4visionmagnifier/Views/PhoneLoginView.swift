//
//  PhoneLoginView.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 11/21/25.
//

import SwiftUI

struct PhoneLoginView: View {
    @StateObject var stackHandler = NavigationStackHandler()

    var body: some View {
        NavigationStack(path: $stackHandler.path) {
            Group {
                EnterPhoneNumberView()
            }
            .navigationDestination(for: NavigationDestinations.self) { destination in
                destination.asView
            }
        }
        .environmentObject(stackHandler)
    }
}

struct EnterPhoneNumberView: View {
    @State private var phoneNumber: String = ""
    @State private var errorMessage: String?
    @State private var isLoading = false
    @EnvironmentObject var stackHandler: NavigationStackHandler

    var body: some View {
        VStack(spacing: 24) {
            Text("Create an Account to Complete Activation")
                .font(.largeTitle)
                .bold()
            
            Text("In order to activate the software, we need to verify your phone number.")

            VStack(alignment: .leading) {
                Text("Phone Number")
                    .font(.headline)
                
                TextField("e.g. +1 555 123 4567", text: $phoneNumber)
                    .keyboardType(.phonePad)
                    .textContentType(.telephoneNumber)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
            
            Button(action: sendCode) {
                label("Send Verification Code")
            }
            .disabled(phoneNumber.isEmpty)
            if let error = errorMessage {
                Text(error)
                    .foregroundColor(.red)
            }
            
            Spacer()
        }
        .padding()
        
    }
    
    // MARK: - UI Helper
    @ViewBuilder
    private func label(_ text: String) -> some View {
        if isLoading {
            ProgressView()
        } else {
            Text(text)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(12)
        }
    }
    
    // MARK: - Actions
    private func sendCode() {
        isLoading = true
        errorMessage = nil
        AuthHandler.shared.getVerificationCode(phoneNumber: phoneNumber)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            isLoading = false
            print("PUSHING VIEW! verify phone number")
            stackHandler.pushView(.verifyPhoneNumber)
        }
    }
}

struct VerifyPhoneNumberView: View {
    @State private var errorMessage: String?
    @State private var verificationCode: String = ""
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 24) {
            Text("Login With Phone")
                .font(.largeTitle)
                .bold()
            
            VStack(alignment: .leading) {
                Text("Verification Code")
                    .font(.headline)
                
                TextField("123456", text: $verificationCode)
                    .keyboardType(.numberPad)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
            
            Button(action: verifyCode) {
                label("Verify & Login")
            }
            .disabled(verificationCode.isEmpty)
            
            if let error = errorMessage {
                Text(error)
                    .foregroundColor(.red)
            }
            
            Spacer()
        }
        .padding()
    }
    
    private func verifyCode() {
        isLoading = true
        errorMessage = nil
        Task {
            let success = await AuthHandler.shared.attemptSignIn(verificationCode: verificationCode)
            print("sign in was successful \(success)")
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                isLoading = false
            }
        }
    }
    
    
    // MARK: - UI Helper
    @ViewBuilder
    private func label(_ text: String) -> some View {
        if isLoading {
            ProgressView()
        } else {
            Text(text)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(12)
        }
    }
}
    

