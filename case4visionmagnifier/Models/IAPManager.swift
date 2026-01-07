//
//  IAPManager.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 1/2/26.
//


import Foundation
import StoreKit

@MainActor
final class IAPManager: ObservableObject {
    static let shared = IAPManager()

    private let productID = "qplusplus_full_license"

    @Published private(set) var product: Product?
    @Published private(set) var isPurchased: Bool = false
    @Published private(set) var isLoading: Bool = false
    @Published internal(set) var lastErrorMessage: String?

    private var updatesTask: Task<Void, Never>?

    private init() {
        // Start listening immediately (important)
        updatesTask = listenForTransactions()

        Task {
            await refreshPurchasedState()
            await loadProduct()
        }
    }

    deinit {
        updatesTask?.cancel()
    }

    func loadProduct() async {
        isLoading = true
        defer { isLoading = false }

        do {
            let products = try await Product.products(for: [productID])
            self.product = products.first
            if self.product == nil {
                self.lastErrorMessage = "Product not found. Check the product ID and App Store Connect configuration."
            }
        } catch {
            self.lastErrorMessage = "Failed to load product: \(error.localizedDescription)"
        }
    }

    func purchase() async {
        guard let product else {
            lastErrorMessage = "Product not loaded."
            return
        }

        isLoading = true
        defer { isLoading = false }

        do {
            let result = try await product.purchase()

            switch result {
            case .success(let verificationResult):
                let transaction = try verify(verificationResult)
                await handleVerifiedTransaction(transaction)

            case .userCancelled:
                // user backed out; no need to show an error
                break

            case .pending:
                // awaiting approval (e.g., Ask to Buy)
                lastErrorMessage = "Purchase pending approval."
                break

            @unknown default:
                lastErrorMessage = "Unknown purchase result."
            }
        } catch {
            lastErrorMessage = "Purchase failed: \(error.localizedDescription)"
        }
    }

    func restorePurchases() async {
        isLoading = true
        defer { isLoading = false }

        // In StoreKit 2, restoring is essentially: sync with App Store + check current entitlements
        do {
            try await AppStore.sync()
            await refreshPurchasedState()
        } catch {
            lastErrorMessage = "Restore failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Internals

    private func listenForTransactions() -> Task<Void, Never> {
        Task.detached(priority: .background) { [weak self] in
            guard let self else { return }
            for await result in Transaction.updates {
                do {
                    let transaction = try await self.verify(result)
                    await self.handleVerifiedTransaction(transaction)
                } catch {
                    await MainActor.run {
                        self.lastErrorMessage = "Transaction verification failed."
                    }
                }
            }
        }
    }

    internal func refreshPurchasedState() async {
        // Current entitlements = what the user currently owns (restored automatically across devices)
        var owned = false
        for await result in Transaction.currentEntitlements {
            if let transaction = try? verify(result),
               transaction.productID == productID {
                owned = true
                break
            }
        }
        print("isPurchased = \(owned)")
        isPurchased = owned
    }

    private func handleVerifiedTransaction(_ transaction: Transaction) async {
        // Mark as purchased if it’s our product
        if transaction.productID == productID {
            print("isPurchased is true")
            isPurchased = true
        }

        // Always finish after granting entitlement
        await transaction.finish()
    }

    private func verify<T>(_ result: VerificationResult<T>) throws -> T {
        switch result {
        case .verified(let signedType):
            return signedType
        case .unverified:
            throw StoreKitError.notEntitled
        }
    }
}
