//
//  PurchaseView.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 1/2/26.
//

import SwiftUI
import StoreKit

struct PurchaseView: View {
    @StateObject private var iap = IAPManager.shared
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(spacing: 16) {
            Text("CloseUp Full License")
                .font(.title)
                .bold()

            if iap.isPurchased {
                Text("✅ Purchased")
                    .font(.headline)
            } else {
                if let product = iap.product {
                    Text(product.displayPrice)
                        .font(.headline)

                    Button {
                        Task { await iap.purchase() }
                    } label: {
                        Text("Buy")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(iap.isLoading)

                    // ✅ NEW: Redeem Offer/Promo Code via Apple's system sheet
                    Button {
                        Task { await redeemOfferCode() }
                    } label: {
                        Text("Redeem Offer Code")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(iap.isLoading)

                    Button {
                        Task { await iap.restorePurchases() }
                    } label: {
                        Text("Restore Purchases")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(iap.isLoading)

                } else {
                    ProgressView("Loading…")
                        .task { await iap.loadProduct() }
                }
            }

            if let msg = iap.lastErrorMessage {
                Text(msg)
                    .foregroundStyle(.red)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        // Helpful: when user returns from redemption sheet, refresh entitlements
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .active {
                Task { await iap.refreshPurchasedStateForUI() }
            }
        }
    }

    private func redeemOfferCode() async {
        // Clear any prior message
        iap.clearErrorForUI()

        do {
            guard let scene = UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene })
                .first(where: { $0.activationState == .foregroundActive }) else {
                await MainActor.run {
                    iap.lastErrorMessage = "Unable to access app window."
                }
                return
            }
            // Presents Apple’s system code redemption UI
            try await AppStore.presentOfferCodeRedeemSheet(in: scene)

            // After dismissal, refresh entitlements (the user might have redeemed successfully)
            await iap.refreshPurchasedStateForUI()
        } catch {
            // The API can throw (e.g., unsupported environment). Keep message user-friendly.
            await MainActor.run {
                iap.setErrorForUI("Could not present code redemption sheet: \(error.localizedDescription)")
            }
        }
    }
}

extension IAPManager {
    // Safe wrappers for the View layer (keeps internals private)
    func refreshPurchasedStateForUI() async {
        await refreshPurchasedState()
    }

    func clearErrorForUI() {
        lastErrorMessage = nil
    }

    func setErrorForUI(_ message: String) {
        lastErrorMessage = message
    }
}


