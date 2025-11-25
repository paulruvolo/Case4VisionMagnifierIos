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
    public static let testSignInFlow = false
    /// Handle to Firebase authentication
    private var firebaseAuth: Auth?
    var activationCode: String?

    @Published private(set) var currentUID: String? = nil
    @Published private(set) var isActivated: Bool = false
    
    func createAuthHandle() {
        firebaseAuth = Auth.auth()
        currentUID = firebaseAuth?.currentUser?.uid
        createAuthListener()
        if Self.testSignInFlow {
            try! Auth.auth().signOut()
        }
        checkForActivation(uid: currentUID)
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
                if !self.isActivated, let currentUID = self.currentUID {
                    self.checkForActivation(uid: currentUID)
                }
                print("currentUID \(self.currentUID ?? "nil")")
            }
        }
    }
    
    private func checkForActivation(uid: String?) {
        if ActivationStore.isActivated() {
            self.isActivated = true
            return
        }
        // check the database
        if let uid {
            Firestore.firestore().collection("users").document(uid).getDocument() { document, error in
                if let error {
                    print("error getting document: \(error)")
                    return
                }
                guard let data = document?.data() else {
                    print("no document found")
                    return
                }
                // as long as there is a value there we count it as valid
                if data["activation"] != nil {
                    print("activating")
                    self.isActivated = true
                }
            }
        }
        
    }
    
    private init() {
        currentUID = nil
        isActivated = ActivationStore.isActivated()
    }
    
    func recordActivationInKeyChain() {
        ActivationStore.setActivated(true)
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
