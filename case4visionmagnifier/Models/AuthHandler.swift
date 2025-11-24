//
//  AuthHandler.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 11/20/25.
//

import FirebaseAuth
import FirebaseFirestore

class AuthHandler: ObservableObject {
    static var shared = AuthHandler()
    private var lastVerificationID: String?
    public static let testSignInFlow = true
    /// Handle to Firebase authentication
    private var firebaseAuth: Auth?
    var activationCode: String?

    @Published private(set) var currentUID: String? = nil
    
    func createAuthHandle() {
       firebaseAuth = Auth.auth()
        currentUID = firebaseAuth?.currentUser?.uid
        createAuthListener()
        if Self.testSignInFlow {
            try! Auth.auth().signOut()
        }
    }
    
    func getVerificationCode(phoneNumber: String) {
        PhoneAuthProvider.provider()
          .verifyPhoneNumber(phoneNumber, uiDelegate: nil) { verificationID, error in
              if let error = error {
                print("had an error verifying phone number ", error.localizedDescription)
                return
              }
              print("verificationID \(verificationID)")
              self.lastVerificationID = verificationID
          }
    }
    
    /// This function responds to authentication state changes so the information in `currentUID`
    private func createAuthListener() {
        firebaseAuth?.addStateDidChangeListener() { (auth, user) in
            DispatchQueue.main.async {
                self.currentUID = user?.uid
                print("currentUID \(self.currentUID ?? "nil")")
            }
        }
    }
    
    private init() {
        currentUID = nil
    }
    
    func recordActivationInKeyChain() {
        print("recording activation")
    }
    
    func attemptSignIn(verificationCode: String) async->Bool {
        guard let lastVerificationID else {
            return false
        }
        let credential = PhoneAuthProvider.provider().credential(
          withVerificationID: lastVerificationID,
          verificationCode: verificationCode
        )
        return await withCheckedContinuation { continuation in
            Auth.auth().signIn(with: credential) { authResult, error in
                if let authResult {
                    DispatchQueue.main.async {
                        continuation.resume(returning: error == nil)
                    }
                }
            }
        }
    }
    
    func associateCodeWithAccount(code: String) async->Bool {
        // TODO: need to actually validate the code on the backend
        guard let currentUID else {
            return false
        }
        return await withCheckedContinuation { continuation in
            Firestore.firestore().collection("users").document(currentUID).setData(["activation": code]) { err in
                if let err = err {
                    print("Error: \(err)")
                    return continuation.resume(returning: false)
                } else {
                    return continuation.resume(returning: true)
                }
            }
        }
    }
}
