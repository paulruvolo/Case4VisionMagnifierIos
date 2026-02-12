//
//  KeychainHelper.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 11/24/25.
//


import Foundation
import Security

struct KeychainHelper {
    static let shared = KeychainHelper()

    private init() {}

    func set(_ value: Data,
             forKey key: String,
             service: String = Bundle.main.bundleIdentifier ?? "case4vision-keychain") {

        let query: [String: Any] = [
            kSecClass as String            : kSecClassGenericPassword,
            kSecAttrService as String      : service,
            kSecAttrAccount as String      : key
        ]

        let attributes: [String: Any] = [
            kSecValueData as String        : value,
            // Adjust accessibility as needed; this is common:
            kSecAttrAccessible as String   : kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        // Try update first
        let status = SecItemUpdate(query as CFDictionary,
                                   attributes as CFDictionary)

        if status == errSecItemNotFound {
            // If not found, add it
            var addQuery = query
            addQuery.merge(attributes) { (_, new) in new }
            SecItemAdd(addQuery as CFDictionary, nil)
        }
    }

    func get(forKey key: String,
             service: String = Bundle.main.bundleIdentifier ?? "case4vision-keychain") -> Data? {

        let query: [String: Any] = [
            kSecClass as String            : kSecClassGenericPassword,
            kSecAttrService as String      : service,
            kSecAttrAccount as String      : key,
            kSecReturnData as String       : kCFBooleanTrue as Any,
            kSecMatchLimit as String       : kSecMatchLimitOne
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess else { return nil }
        return item as? Data
    }

    func delete(forKey key: String,
                service: String = Bundle.main.bundleIdentifier ?? "case4vision-keychain") {

        let query: [String: Any] = [
            kSecClass as String       : kSecClassGenericPassword,
            kSecAttrService as String : service,
            kSecAttrAccount as String : key
        ]
        SecItemDelete(query as CFDictionary)
    }
}

struct ActivationStore {
    private static let key = "activation_flag"

    static func setActivated(_ activated: Bool) {
        let data = Data([activated ? 1 : 0])
        KeychainHelper.shared.set(data, forKey: key)
        print("SETTING KEY CHAIN DATA")
    }

    static func isActivated() -> Bool {
        guard let data = KeychainHelper.shared.get(forKey: key),
              let byte = data.first else {
            return false
        }
        return byte == 1
    }

    static func clearActivation() {
        KeychainHelper.shared.delete(forKey: key)
    }
}


struct TrialStore {
    private static let key = "trial_start_timestamp"

    /// Call once when you want to start the trial.
    static func startTrialIfNeeded() {
        if getTrialStartDate() != nil {
            return // already started
        }
        let now = Date()
        let seconds = Double(now.timeIntervalSince1970)
        let data = withUnsafeBytes(of: seconds) { Data($0) }
        print("starting trial if needed \(seconds)")
        KeychainHelper.shared.set(data, forKey: key)
    }

    /// Returns the recorded trial start date, if any.
    static func getTrialStartDate() -> Date? {
        guard let data = KeychainHelper.shared.get(forKey: key),
              data.count == MemoryLayout<UInt64>.size else {
            return nil
        }
        let seconds = data.withUnsafeBytes { $0.load(as: Double.self) }
        return Date(timeIntervalSince1970: seconds)
    }

    /// True if we have a trial and it's older than `days` days.
    static func isTrialExpired() -> Bool {
        guard let start = getTrialStartDate() else {
            // If no start date, treat as not started (or expired, your choice)
            return false
        }
        let elapsed = Date().timeIntervalSince(start)
        let limit = TimeInterval(case4visionmagnifierApp.trialLimit * 24 * 60 * 60)
        return elapsed >= limit
    }
    
    static func trialDaysLeft() -> Int {
        guard let start = getTrialStartDate() else {
            return 0
        }
        let elapsed = Date().timeIntervalSince(start)
        let limit: TimeInterval = 24 * 60 * 60 * Double(case4visionmagnifierApp.trialLimit)
        let remaining = (limit - elapsed)
        return Int(round(remaining / 24.0 / 60.0 / 60.0))
    }

    /// For debugging / reset
    static func clearTrial() {
        KeychainHelper.shared.delete(forKey: key)
    }
}
