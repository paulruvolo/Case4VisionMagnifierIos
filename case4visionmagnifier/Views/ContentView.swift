//
//  ContentView.swift
//  case4visionmagnifier
//
//  Created by Paul Ruvolo on 8/13/25.
//

import SwiftUI

struct ContentView: View {
    @State var showLandscape: Bool = false
    var body: some View {
        VStack {
            Button("Magnifier") {
                showLandscape.toggle()
            }
        }
        .padding()
        .fullScreenCover(isPresented: $showLandscape) {
            RearWideCameraView()
        }
    }
}

#Preview {
    ContentView()
}
