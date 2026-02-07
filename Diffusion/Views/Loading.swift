//
//  Loading.swift
//  Diffusion
//
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import SwiftUI
import Combine

func iosModel() -> ModelInfo {
    guard deviceSupportsQuantization else { return ModelInfo.v21Base }
    //if deviceHas6GBOrMore { return ModelInfo.xlmbpChunked }
    if deviceHas6GBOrMore { return ModelInfo.xlmbpChunked }
    return ModelInfo.v21Palettized
}

// Add this helper
func checkpointShortName(for model: ModelInfo) -> String {
    if model.modelId == ModelInfo.xlmbpChunked.modelId { return "xlmbpChunked" }
    if model.modelId == ModelInfo.v21Palettized.modelId { return "v21Palettized" }
    if model.modelId == ModelInfo.v21Base.modelId { return "v21Base" }
    return model.modelId // fallback
}

struct LoadingView: View {

    @StateObject var generation = GenerationContext()

    @State private var preparationPhase = "Preparing diffusion engine"
    @State private var preparationDetail = "Checking local model files"
    @State private var downloadProgress: Double? = nil
    
    enum CurrentView {
        case loading
        case textToImage
        case error(String)
    }
    @State private var currentView: CurrentView = .loading
    
    @State private var stateSubscriber: Cancellable?

    var body: some View {
        VStack {
            switch currentView {
            case .textToImage: TextToImage().transition(.opacity)
            case .error(let message): ErrorPopover(errorMessage: message).transition(.move(edge: .top))
            case .loading:
                BrandedLoadingView(
                    phase: preparationPhase,
                    detail: preparationDetail,
                    progress: downloadProgress
                )
            }
        }
        .animation(.easeIn, value: currentView)
        .environmentObject(generation)
        .onAppear {
            Task.init {
                // ✅ Select model once, and capture its label for the progress UI
                let selectedModel = iosModel()
                let selectedModelName = checkpointShortName(for: selectedModel)

                let loader = PipelineLoader(model: selectedModel)
                stateSubscriber = loader.statePublisher.sink { state in
                    DispatchQueue.main.async {
                        switch state {
                        case .downloading(let progress):
                            preparationPhase = "Downloading \(selectedModelName)"
                            preparationDetail = "First launch downloads model files to your device"
                            downloadProgress = progress
                        case .uncompressing:
                            preparationPhase = "Uncompressing \(selectedModelName)"
                            preparationDetail = "Optimizing model files for on-device inference"
                            downloadProgress = nil
                        case .readyOnDisk:
                            preparationPhase = "Loading \(selectedModelName)"
                            preparationDetail = "Warming up the generation pipeline"
                            downloadProgress = nil
                        default:
                            break
                        }
                    }
                }
                do {
                    generation.pipeline = try await loader.prepare()
                    self.currentView = .textToImage
                } catch {
                    self.currentView = .error("Could not load model, error: \(error)")
                }
            }
        }
    }
}

// Required by .animation
extension LoadingView.CurrentView: Equatable {}

struct BrandedLoadingView: View {
    var phase: String
    var detail: String
    var progress: Double?

    private var clampedProgress: Double {
        min(max(progress ?? 0, 0), 1)
    }

    var body: some View {
        GeometryReader { proxy in
            let boxSide = min(proxy.size.width, proxy.size.height) * 0.5
            let cornerRadius = boxSide * 0.12
            let iconSide = boxSide * 0.82

            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.10, green: 0.12, blue: 0.23),
                        Color(red: 0.07, green: 0.20, blue: 0.33)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                VStack {
                    Spacer(minLength: 0)

                    ZStack {
                        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                            .fill(.white.opacity(0.14))
                        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                            .stroke(.white.opacity(0.22), lineWidth: 1.2)
                        Image("LaunchIcon")
                            .resizable()
                            .scaledToFit()
                            .frame(width: iconSide, height: iconSide)
                    }
                    .frame(width: boxSide, height: boxSide)
                    .shadow(color: .black.opacity(0.25), radius: 20, x: 0, y: 10)

                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                VStack(spacing: 10) {
                    Text(phase)
                        .font(.headline)
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                    Text(detail)
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.85))
                        .multilineTextAlignment(.center)

                    if progress != nil {
                        ProgressView(value: clampedProgress, total: 1)
                            .tint(.white)
                            .progressViewStyle(.linear)
                            .padding(.top, 4)
                        Text("\(Int(clampedProgress * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.86))
                    } else {
                        HStack(spacing: 10) {
                            ProgressView().tint(.white)
                            Text("Working…")
                                .foregroundStyle(.white.opacity(0.9))
                                .font(.footnote)
                        }
                        .padding(.top, 4)
                    }
                }
                .padding(18)
                .background(.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(.white.opacity(0.18), lineWidth: 1)
                )
                .frame(maxWidth: min(proxy.size.width * 0.9, 560))
                .padding(.horizontal, 22)
                .padding(.bottom, 26)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
            }
        }
    }
}

struct ErrorPopover: View {
    var errorMessage: String

    var body: some View {
        Text(errorMessage)
            .font(.headline)
            .padding()
            .foregroundColor(.red)
            .background(Color.white)
            .cornerRadius(8)
            .shadow(color: Color.black.opacity(0.2), radius: 8, x: 0, y: 4)
    }
}

struct LoadingView_Previews: PreviewProvider {
    static var previews: some View {
        LoadingView()
    }
}
