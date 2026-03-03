//
//  ControlsView.swift
//  Diffusion-macOS
//

import SwiftUI
import UniformTypeIdentifiers

private enum MacPathPickerTarget: Equatable {
    case transformer
    case vae
    case embeddings
    case initialLatent
}

enum PipelineState {
    case downloading(Double)
    case uncompressing
    case loading
    case ready
    case failed(Error)
}

private struct PathSelectionError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private struct BootstrapContextError: LocalizedError {
    let summary: String
    let transformerPath: String
    let vaePath: String
    let embeddingsPath: String
    let resolutionDetails: [String]
    let underlyingError: Error

    var errorDescription: String? {
        let detailsBlock = resolutionDetails.joined(separator: "\n")
        return """
        \(summary)
        Transformer: \(transformerPath)
        VAE: \(vaePath)
        Embeddings: \(embeddingsPath)
        \(detailsBlock)
        Underlying error: \(underlyingError)
        """
    }
}

@available(macOS 14.0, *)
struct ControlsView: View {
    @EnvironmentObject var generation: GenerationContext

    @State private var pipelineState: PipelineState = .loading
    @State private var seedText: String = String(Settings.shared.seed)
    @State private var bootstrapDone = false
    @State private var activePathPicker: MacPathPickerTarget?
    @State private var pendingPathPicker: MacPathPickerTarget?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("ZImage Checkpoint Test", systemImage: "cpu")
                .font(.headline)
            Text("Path mode: pick Transformer and VAE locations directly. VAE can remain fixed while swapping Transformer checkpoints.")
                .font(.caption)
                .foregroundColor(.secondary)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Button("Select Transformer Path") {
                        pendingPathPicker = .transformer
                        activePathPicker = .transformer
                    }
                    .buttonStyle(.bordered)
                    // if generation.transformerModelPath != nil {
                    //     Button("Use Bundled Transformer") {
                    //         generation.setTransformerModelURL(nil)
                    //         Task { await bootstrapPipeline() }
                    //     }
                    //     .buttonStyle(.bordered)
                    // }
                }

                HStack {
                    Button("Select VAE Path") {
                        pendingPathPicker = .vae
                        activePathPicker = .vae
                    }
                    .buttonStyle(.bordered)
                    // if generation.vaeDecoderPath != nil {
                    //     Button("Use Bundled VAE") {
                    //         generation.setVaeDecoderModelURL(nil)
                    //         Task { await bootstrapPipeline() }
                    //     }
                    //     .buttonStyle(.bordered)
                    // }
                }

                Button("Reload Models") {
                    Task { await bootstrapPipeline() }
                }
                .buttonStyle(.bordered)

                Text("Transformer: \(generation.transformerModelPath.flatMap { URL(fileURLWithPath: $0).lastPathComponent } ?? "Bundled") (\(generation.transformerStageURLs.count) stages)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("VAE: \(generation.vaeDecoderPath.flatMap { URL(fileURLWithPath: $0).lastPathComponent } ?? "Bundled VAEDecoder.mlmodelc")")
                    .font(.caption)
                    .foregroundColor(.secondary)
                // Text("Chosen Transformer Path: \(generation.transformerModelPath ?? "Bundled ZImageTurbo_TransformerBackbone.mlmodelc")")
                //     .font(.caption2)
                //     .foregroundColor(.secondary)
                //     .textSelection(.enabled)
                // Text("Chosen VAE Path: \(generation.vaeDecoderPath ?? "Bundled VAEDecoder.mlmodelc")")
                //     .font(.caption2)
                //     .foregroundColor(.secondary)
                //     .textSelection(.enabled)
                Text("Resolved Transformer Dir: \(generation.transformerModelURL.deletingLastPathComponent().path)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
                Text("Resolved VAE Path: \(generation.vaeDecoderModelURL.path)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }

            runConfigurationSection
            PromptTextField(text: $generation.positivePrompt, isPositivePrompt: true, model: .constant("zimage"))
                .onChange(of: generation.positivePrompt) { _, prompt in
                    Settings.shared.prompt = prompt
                }

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Button("Select Embeddings Path") {
                        pendingPathPicker = .embeddings
                        activePathPicker = .embeddings
                    }
                    .buttonStyle(.bordered)
                    if generation.externalEmbeddingsPath != nil {
                        Button("Clear Embeddings Path") {
                            generation.setExternalEmbeddingsURL(nil)
                            Task { await bootstrapPipeline() }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                Text(embeddingsStatusText())
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Text encoder folder is not used yet; embeddings tensor file is required.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Button("Select Init Latent Path") {
                        pendingPathPicker = .initialLatent
                        activePathPicker = .initialLatent
                    }
                    .buttonStyle(.bordered)
                    if generation.initialLatentPath != nil {
                        Button("Clear Init Latent Path") {
                            generation.setInitialLatentURL(nil)
                        }
                        .buttonStyle(.bordered)
                    }
                }
                Text(initialLatentStatusText())
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Expected format: raw Float32 .bin, shape [1,16,64,64].")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Resolved Init Latent Path: \(generation.initialLatentFileURL?.path ?? "Not set")")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }

            VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Steps")
                            Spacer()
                            Text("\(Int(generation.steps))")
                        .foregroundColor(.secondary)
                }
                Stepper("", value: Binding(
                    get: { Int(generation.steps) },
                    set: {
                        let clamped = max(1, min(50, $0))
                        generation.steps = Double(clamped)
                        Settings.shared.stepCount = Double(clamped)
                    }
                ), in: 1...50)
                .labelsHidden()
            }

            VStack(alignment: .leading, spacing: 8) {
                            HStack {
                    Text("Seed")
                                Spacer()
                    TextField("0", text: $seedText)
                        .multilineTextAlignment(.trailing)
                        .frame(maxWidth: 140)
                }
                Text("Seed 0 means random seed on each run.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .onChange(of: seedText) { _, newValue in
                let filtered = newValue.filter { "0123456789".contains($0) }
                if filtered != newValue {
                    seedText = filtered
                }
                let seed = UInt32(filtered) ?? 0
                generation.seed = seed
                Settings.shared.seed = seed
            }

            Divider()
            StatusView(pipelineState: $pipelineState)
        }
        .padding()
        .onAppear {
            guard !bootstrapDone else { return }
            bootstrapDone = true
            Task {
                await bootstrapPipeline()
            }
        }
        .fileImporter(
            isPresented: Binding(
                get: { activePathPicker != nil },
                set: { if !$0 { activePathPicker = nil } }
            ),
            allowedContentTypes: {
                switch pendingPathPicker ?? activePathPicker {
                case .transformer, .vae:
                    return [.item]
                case .embeddings, .initialLatent:
                    return [.data]
                case .none:
                    return [.item]
                }
            }(),
            allowsMultipleSelection: false
        ) { result in
            let picker = pendingPathPicker ?? activePathPicker
            activePathPicker = nil
            pendingPathPicker = nil
            switch result {
            case .success(let urls):
                guard let first = urls.first else { return }
                switch picker {
                case .transformer:
                    Task { @MainActor in
                        applyModelSelection(url: first, target: .transformer)
                    }
                case .vae:
                    Task { @MainActor in
                        applyModelSelection(url: first, target: .vae)
                    }
                case .embeddings:
                    Task { @MainActor in
                        generation.setExternalEmbeddingsURL(first)
                        await bootstrapPipeline()
                    }
                case .initialLatent:
                    Task { @MainActor in
                        generation.setInitialLatentURL(first)
                    }
                case .none:
                    return
                }
            case .failure(let error):
                pipelineState = .failed(error)
            }
        }
    }

    @MainActor
    private func bootstrapPipeline() async {
        pipelineState = .loading
        do {
            let stageURLs = generation.transformerStageURLs
            let vaeURL = generation.vaeDecoderModelURL
            let embeddingsURL = generation.effectiveEmbeddingsURL
            let bootstrap = ZImageBootstrapConfig(
                transformerStageURLs: stageURLs,
                vaeDecoderURL: vaeURL,
                embeddingsURL: embeddingsURL
            )
            let loader = ZImagePipelineLoader(config: bootstrap, computeUnits: generation.computeUnits)
            generation.pipeline = try loader.loadAppPipeline(runSmokeTest: false, smokeSteps: 4, smokeSeed: 42)
            pipelineState = .ready
        } catch {
            pipelineState = .failed(
                BootstrapContextError(
                    summary: "Failed to load pipeline with current resolved paths.",
                    transformerPath: generation.transformerModelURL.path,
                    vaePath: generation.vaeDecoderModelURL.path,
                    embeddingsPath: generation.effectiveEmbeddingsURL.path,
                    resolutionDetails: [
                        "Transformer detail: \(generation.transformerPathResolutionDetail)",
                        "VAE detail: \(generation.vaePathResolutionDetail)",
                        "Embeddings detail: \(generation.embeddingsPathResolutionDetail)"
                    ],
                    underlyingError: error
                )
            )
        }
    }

    @MainActor
    private func applyModelSelection(url: URL, target: MacPathPickerTarget) {
        guard validateSelectedModelDirectory(url) else {
            let modelName = (target == .transformer) ? "Transformer" : "VAE"
            pipelineState = .failed(PathSelectionError(message: "\(modelName) selection must be a readable .mlmodelc directory."))
            return
        }

        switch target {
        case .transformer:
            generation.setTransformerModelURL(url)
            let resolved = generation.transformerModelURL.standardizedFileURL.path
            guard resolved == url.standardizedFileURL.path else {
                pipelineState = .failed(
                    PathSelectionError(
                        message: "Selected Transformer path did not persist. Resolved path is \(resolved)."
                    )
                )
                return
            }
        case .vae:
            generation.setVaeDecoderModelURL(url)
            let resolved = generation.vaeDecoderModelURL.standardizedFileURL.path
            guard resolved == url.standardizedFileURL.path else {
                pipelineState = .failed(
                    PathSelectionError(
                        message: "Selected VAE path did not persist. Resolved path is \(resolved)."
                    )
                )
                return
            }
        case .embeddings:
            return
        case .initialLatent:
            return
        }

        Task { await bootstrapPipeline() }
    }

    private func validateSelectedModelDirectory(_ url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "mlmodelc" else {
            return false
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }
        let metadataPath = url.appending(path: "metadata.json").path
        return FileManager.default.fileExists(atPath: metadataPath)
    }

    private func embeddingsStatusText() -> String {
        if let path = generation.externalEmbeddingsPath, !path.isEmpty {
            return "Using external embeddings: \(URL(fileURLWithPath: path).lastPathComponent)"
        }
        return "Using bundled embeddings file."
    }

    private func initialLatentStatusText() -> String {
        if let path = generation.initialLatentPath, !path.isEmpty {
            return "Using initial latent file: \(URL(fileURLWithPath: path).lastPathComponent)"
        }
        return "No initial latent file selected."
    }

    private var runConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Run Configuration")
                .font(.headline)
            Text("Scheduler: \(generation.scheduler.rawValue)")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("CFG: \(String(format: "%.2f", generation.guidanceScale))")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("Output: 512 x 512 (fixed)")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("Latents: channels=16, size=64 x 64 (fixed)")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("Steps: \(Int(generation.steps)) | Seed: \(generation.seed)")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("Compute Units: \(String(describing: generation.computeUnits))")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}
