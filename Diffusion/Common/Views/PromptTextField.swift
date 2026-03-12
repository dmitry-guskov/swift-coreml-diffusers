//
//  PromptTextField.swift
//  Diffusion-macOS
//
//  Created by Dolmere on 22/06/2023.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import SwiftUI
import Combine

struct PromptTextField: View {
    @State private var tokenCount: Int = 0
    @State var isPositivePrompt: Bool = true

    @Binding var textBinding: String
    @Binding var model: String // kept for call-site compatibility

    private let maxTokenCount = 500
    
    private var textColor: Color {
        switch tokenCount {
        case 0...65:
            return .green
        case 66...75:
            return .orange
        default:
            return .red
        }
    }
    
    // macOS initializer
    init(text: Binding<String>, isPositivePrompt: Bool, model: Binding<String>) {
         _textBinding = text
         self.isPositivePrompt = isPositivePrompt
        _model = model
    }
    
    // iOS initializer
    init(text: Binding<String>, isPositivePrompt: Bool, model: String) {
        _textBinding = text
        self.isPositivePrompt = isPositivePrompt
        _model = .constant(model)
    }

    var body: some View {
        VStack {
            #if os(macOS)
            TextField(isPositivePrompt ? "Positive prompt" : "Negative Prompt", text: $textBinding,
                      axis: .vertical)
                .lineLimit(20)
                .textFieldStyle(.squareBorder)
                .listRowInsets(EdgeInsets(top: 0, leading: -20, bottom: 0, trailing: 20))
                .foregroundColor(textColor == .green ? .primary : textColor)
                .frame(minHeight: 30)
            HStack {
                Spacer()
                if !textBinding.isEmpty {
                    Text("\(tokenCount)")
                        .foregroundColor(textColor)
                    Text(" / \(maxTokenCount)")
                }
            }
            .onReceive(Just(textBinding)) { text in
                updateTokenCount(newText: text)
            }
            .font(.caption)
            #else
            TextField("Prompt", text: $textBinding, axis: .vertical)
                .lineLimit(20)
                .listRowInsets(EdgeInsets(top: 0, leading: -20, bottom: 0, trailing: 20))
                .foregroundColor(textColor == .green ? .primary : textColor)
                .frame(minHeight: 30)
            HStack {
                if !textBinding.isEmpty {
                    Text("\(tokenCount)")
                        .foregroundColor(textColor)
                    Text(" / \(maxTokenCount)")
                }
                Spacer()
            }
            .onReceive(Just(textBinding)) { text in
                updateTokenCount(newText: text)
            }
            .font(.caption)
            #endif
        }
        .onChange(of: model) { model in
            updateTokenCount(newText: textBinding)
        }
        .onAppear {
            updateTokenCount(newText: textBinding)
        }
    }

    private func updateTokenCount(newText: String) {
        let tokens = newText
            .split(whereSeparator: { $0.isWhitespace || $0.isNewline })
            .count
        DispatchQueue.main.async {
            self.tokenCount = tokens
        }
    }
}
