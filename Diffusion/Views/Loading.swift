//
//  Loading.swift
//  Diffusion
//
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import SwiftUI

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
                do {
                    preparationPhase = "Validating resources"
                    preparationDetail = "Checking Transformer, VAE, and embeddings paths"
                    downloadProgress = nil

                    preparationPhase = "Loading models"
                    preparationDetail = "Initializing Transformer and VAE resources"

                    if #available(iOS 18.0, macOS 14.0, *) {
                        let bootstrap = ZImageBootstrapConfig(
                            transformerStageURLs: generation.transformerStageURLs,
                            vaeDecoderURL: generation.vaeDecoderModelURL,
                            embeddingsURL: generation.effectiveEmbeddingsURL,
                            loraURL: generation.effectiveLoRAURL
                        )
                        let loader = ZImagePipelineLoader(config: bootstrap, computeUnits: generation.computeUnits)
                        generation.pipeline = try loader.loadAppPipeline(runSmokeTest: false, smokeSteps: 4, smokeSeed: 42)
                        preparationPhase = "Ready"
                        preparationDetail = "Pipeline loaded successfully"
                    } else {
                        throw "ZImage test mode requires iOS 17 / macOS 14"
                    }
                    self.currentView = .textToImage
                } catch {
                    print("[Loading] Model loading failed: \(error). Proceeding to main view for configuration.")
                    self.currentView = .textToImage
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

            VStack(spacing: 26) {
                Spacer(minLength: 24)

                VStack(spacing: 14) {
                    Image("LaunchIcon")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 108, height: 108)
                        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                        .shadow(color: .black.opacity(0.25), radius: 16, x: 0, y: 8)

                    Text("Glam Text-to-Image")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                    Text("On-device image generation")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.82))
                }
                .multilineTextAlignment(.center)

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

                Spacer(minLength: 10)
            }
            .padding(22)
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
