//
//  ZImagePipeline.swift
//  stable-diffusion
//
//  Created by Dmitry Guskov on 10.02.2026.
//

import Accelerate
import CoreGraphics
import CoreML
import Foundation

// MARK: - Scheduler enum

/// Schedulers compatible with ZImagePipeline
public enum ZImageSchedulerType {
    /// Scheduler for rectified flow based diffusion transformer
    case discreteFlowScheduler
}

// MARK: - Configuration

/// Generation configuration for ZImagePipeline
@available(iOS 17.0, macOS 14.0, *)
public struct ZImageConfiguration {
    /// URL to a file containing pre-computed text embeddings
    /// (MLShapedArray-compatible tensor, e.g. [1, seq_len, cap_feat_dim])
    public var embeddingsURL: URL

    /// Number of denoising steps
    public var stepCount: Int

    /// Random seed
    public var seed: UInt32

    /// Scheduler type (only discrete flow for Z-Image)
    public var schedulerType: ZImageSchedulerType = .discreteFlowScheduler

    /// Timestep shift for the flow scheduler (Python DEFAULT_SCHEDULER_SHIFT = 3.0)
    public var schedulerTimestepShift: Float = 3.0

    /// RNG type for latent noise generation
    public var rngType: StableDiffusionRNG = .torchRNG

    /// Whether to pass denoised (rather than noisy) intermediates to the progress handler
    public var useDenoisedIntermediates: Bool = true

    /// Optional LoRA adapter URL to apply to the DiT model
    public var loraURL: URL? = nil

    /// Optional injected initial latent tensor bytes (Float32, row-major)
    public var initialLatentData: Data? = nil

    /// Shape for `initialLatentData` (must match [1, 16, 64, 64])
    public var initialLatentShape: [Int]? = nil

    /// Enable writing debug checkpoint tensors during sampling
    public var debugEnabled: Bool = false

    /// Directory where debug tensors and metadata are exported
    public var debugOutputDirectory: URL? = nil

    /// Save initial latent before the denoising loop
    public var debugSaveInitialLatent: Bool = false

    /// Save DiT output tensor at each step before scheduler update
    public var debugSaveDitOutputEachStep: Bool = false

    /// Save latent tensor at each step after scheduler update
    public var debugSaveLatentAfterSchedulerEachStep: Bool = false

    /// Skip VAE decode and export final latent instead
    public var debugSkipVaeDecode: Bool = false

    /// Always 1 — single image, on-device generation
    public var imageCount: Int { 1 }

    public init(
        embeddingsURL: URL,
        stepCount: Int = 4,
        seed: UInt32 = 0,
        loraURL: URL? = nil
    ) {
        self.embeddingsURL = embeddingsURL
        self.stepCount = stepCount
        self.seed = seed
        self.loraURL = loraURL
    }
}

// MARK: - Protocol

@available(iOS 17.0, macOS 14.0, *)
public protocol ZImagePipelineProtocol: ResourceManaging {
    var canSafetyCheck: Bool { get }

    func generateImages(
        configuration config: ZImageConfiguration,
        progressHandler: (ZImageProgress) -> Bool
    ) throws -> [CGImage?]

    func decodeToImages(
        _ latents: [MLShapedArray<Float32>],
        configuration config: ZImageConfiguration
    ) throws -> [CGImage?]
}

@available(iOS 17.0, macOS 14.0, *)
public extension ZImagePipelineProtocol {
    var canSafetyCheck: Bool { false }
}

// MARK: - Pipeline

@available(iOS 17.0, macOS 14.0, *)
public struct ZImagePipeline: ZImagePipelineProtocol {
    public enum Error: Swift.Error, LocalizedError {
        case unexpectedLatentShape(actual: [Int], expected: [Int])
        case invalidEmbeddingsByteCount(actual: Int, expected: Int)
        case unexpectedEmbeddingsShape(actual: [Int], expected: [Int])
        case invalidInitialLatentShape(actual: [Int], expected: [Int])
        case invalidInitialLatentByteCount(actual: Int, expected: Int)
        case invalidInitialLatentConfiguration

        public var errorDescription: String? {
            switch self {
            case .unexpectedLatentShape(let actual, let expected):
                return "Unexpected latent shape. Expected \(expected), got \(actual)."
            case .invalidEmbeddingsByteCount(let actual, let expected):
                return "Invalid embeddings byte count. Expected \(expected), got \(actual)."
            case .unexpectedEmbeddingsShape(let actual, let expected):
                return "Unexpected embeddings shape. Expected \(expected), got \(actual)."
            case .invalidInitialLatentShape(let actual, let expected):
                return "Invalid initial latent shape. Expected \(expected), got \(actual)."
            case .invalidInitialLatentByteCount(let actual, let expected):
                return "Invalid initial latent byte count. Expected \(expected), got \(actual)."
            case .invalidInitialLatentConfiguration:
                return "Initial latent configuration is incomplete. Provide both initialLatentData and initialLatentShape."
            }
        }
    }

    private let expectedLatentShape = [1, 16, 64, 64]
    private let expectedEmbeddingShape = [1, 77, 2560]
    private var expectedEmbeddingFloatCount: Int { expectedEmbeddingShape[0] * expectedEmbeddingShape[1] * expectedEmbeddingShape[2] }
    private var expectedEmbeddingByteCount: Int { expectedEmbeddingFloatCount * MemoryLayout<Float32>.size }
    private var expectedLatentFloatCount: Int { expectedLatentShape.reduce(1, *) }
    private var expectedLatentByteCount: Int { expectedLatentFloatCount * MemoryLayout<Float32>.size }

    var dit: Dit
    var vae: AutoencoderKLZImage
    var reduceMemory: Bool = false

    public init(
        dit: Dit,
        vae: AutoencoderKLZImage,
        reduceMemory: Bool = false
    ) {
        self.dit = dit
        self.vae = vae
        self.reduceMemory = reduceMemory
    }

    // MARK: - ResourceManaging

    public func loadResources() throws {
        logMemory("ZImagePipeline.loadResources.start")
        if reduceMemory {
            print("[ZImagePipeline] loadResources: reduceMemory=true, using prewarm")
            try prewarmResources()
        } else {
            print("[ZImagePipeline] loadResources: reduceMemory=false, loading both models")
            logMemory("ZImagePipeline.loadResources.beforeDit")
            try dit.loadResources()
            logMemory("ZImagePipeline.loadResources.afterDit")
            try vae.loadResources()
            logMemory("ZImagePipeline.loadResources.afterVae")
        }
        logMemory("ZImagePipeline.loadResources.complete")
    }

    public func unloadResources() {
        logMemory("ZImagePipeline.unloadResources.start")
        dit.unloadResources()
        logMemory("ZImagePipeline.unloadResources.afterDit")
        vae.unloadResources()
        logMemory("ZImagePipeline.unloadResources.complete")
    }

    public func prewarmResources() throws {
        logMemory("ZImagePipeline.prewarmResources.start")
        
        // Prewarm DiT in its own autoreleasepool to ensure memory release
        print("[ZImagePipeline] Prewarming DiT...")
        try autoreleasepool {
            try dit.prewarmResources()
        }
        logMemory("ZImagePipeline.prewarmResources.afterDit")
        
        // Prewarm VAE separately after DiT memory is released
        print("[ZImagePipeline] Prewarming VAE...")
        try autoreleasepool {
            try vae.prewarmResources()
        }
        logMemory("ZImagePipeline.prewarmResources.complete")
    }
    
    private func logMemory(_ checkpoint: String) {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let usedMB = Double(info.resident_size) / 1_048_576
            print("[Memory] \(checkpoint): \(String(format: "%.1f", usedMB)) MB")
        }
    }

    // MARK: - Image generation

    public func generateImages(
        configuration config: ZImageConfiguration,
        progressHandler: (ZImageProgress) -> Bool = { _ in true }
    ) throws -> [CGImage?] {
        logMemory("ZImagePipeline.generateImages.start")
        let debugDirectory = try prepareDebugDirectoryIfNeeded(config: config)

        // Load pre-computed text embeddings from file
        logMemory("ZImagePipeline.generateImages.beforeLoadEmbeddings")
        let hiddenStates = try loadEmbeddings(from: config.embeddingsURL)
        logMemory("ZImagePipeline.generateImages.afterLoadEmbeddings")
        guard hiddenStates.shape == expectedEmbeddingShape else {
            throw Error.unexpectedEmbeddingsShape(actual: hiddenStates.shape, expected: expectedEmbeddingShape)
        }

        // Setup scheduler (single instance — one image)
        let scheduler: Scheduler = ZimageDiscreteFlowScheduler(
            stepCount: config.stepCount,
            timeStepShift: config.schedulerTimestepShift
        )

        // Generate random initial latent noise
        var latent = try generateLatentSample(configuration: config, scheduler: scheduler)
        logMemory("ZImagePipeline.generateImages.afterLatentSample")
        if config.debugEnabled && config.debugSaveInitialLatent {
            try writeDebugTensor(
                latent,
                kind: "latent_initial",
                timestep: nil,
                stepIndex: nil,
                seed: config.seed,
                outputDirectory: debugDirectory
            )
        }

        // Will hold the denoised intermediate for the decoder
        var denoisedLatent = latent

        // De-noising loop with autoreleasepool for memory optimization
        let timeSteps: [Int] = scheduler.calculateTimesteps(strength: nil)
        for (step, t) in timeSteps.enumerated() {
            let shouldContinue: Bool = try autoreleasepool {
                if config.debugEnabled {
                    logFiniteStats(
                        latent,
                        label: "latent_before_dit",
                        timestep: t,
                        stepIndex: step
                    )
                }

                // Predict noise residual conditioned on text embeddings
                let noise = try dit.predictNoise(
                    latents: [latent],
                    timeStep: t,
                    hiddenStates: hiddenStates
                )
                if config.debugEnabled {
                    logFiniteStats(
                        noise[0],
                        label: "dit_output",
                        timestep: t,
                        stepIndex: step
                    )
                }
                if config.debugEnabled && config.debugSaveDitOutputEachStep {
                    try writeDebugTensor(
                        noise[0],
                        kind: "dit_output",
                        timestep: t,
                        stepIndex: step,
                        seed: config.seed,
                        outputDirectory: debugDirectory
                    )
                }

                // Scheduler step: compute previous latent sample
                latent = scheduler.step(
                    output: noise[0],
                    timeStep: t,
                    sample: latent
                )
                if config.debugEnabled {
                    logFiniteStats(
                        latent,
                        label: "latent_after_scheduler",
                        timestep: t,
                        stepIndex: step
                    )
                }
                if config.debugEnabled && config.debugSaveLatentAfterSchedulerEachStep {
                    try writeDebugTensor(
                        latent,
                        kind: "latent_after_step",
                        timestep: t,
                        stepIndex: step,
                        seed: config.seed,
                        outputDirectory: debugDirectory
                    )
                }

                denoisedLatent = scheduler.modelOutputs.last ?? latent

                let currentSample = config.useDenoisedIntermediates ? denoisedLatent : latent

                // Report progress
                let progress = ZImageProgress(
                    pipeline: self,
                    step: step,
                    stepCount: timeSteps.count,
                    currentLatentSample: currentSample
                )
                return progressHandler(progress)
            }
            if !shouldContinue {
                return []
            }
        }

        logMemory("ZImagePipeline.generateImages.afterDenoisingLoop")
        
        if reduceMemory {
            logMemory("ZImagePipeline.generateImages.beforeDitUnload")
            dit.unloadResources()
            logMemory("ZImagePipeline.generateImages.afterDitUnload")
        }

        if config.debugEnabled && config.debugSkipVaeDecode {
            try writeDebugTensor(
                denoisedLatent,
                kind: "latent_final_pre_vae",
                timestep: nil,
                stepIndex: timeSteps.isEmpty ? nil : (timeSteps.count - 1),
                seed: config.seed,
                outputDirectory: debugDirectory
            )
            return []
        }

        // Decode the final latent to an image
        logMemory("ZImagePipeline.generateImages.beforeDecode")
        let images = try decodeToImages([denoisedLatent], configuration: config)
        logMemory("ZImagePipeline.generateImages.afterDecode")
        return images
    }

    // MARK: - Latent generation

    func generateLatentSample(
        configuration config: ZImageConfiguration,
        scheduler: Scheduler
    ) throws -> MLShapedArray<Float32> {
        if (config.initialLatentData == nil) != (config.initialLatentShape == nil) {
            throw Error.invalidInitialLatentConfiguration
        }
        if let initialLatentData = config.initialLatentData,
           let initialLatentShape = config.initialLatentShape {
            guard initialLatentShape == expectedLatentShape else {
                throw Error.invalidInitialLatentShape(actual: initialLatentShape, expected: expectedLatentShape)
            }
            guard initialLatentData.count == expectedLatentByteCount else {
                throw Error.invalidInitialLatentByteCount(actual: initialLatentData.count, expected: expectedLatentByteCount)
            }
            let floats: [Float32] = initialLatentData.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Float32.self))
            }
            return MLShapedArray<Float32>(scalars: floats, shape: expectedLatentShape)
        }

        // Use expected shape directly to avoid triggering early model load
        // The model shape validation happens lazily during first prediction
        let sampleShape = expectedLatentShape

        let stdev = scheduler.initNoiseSigma
        var random = randomSource(from: config.rngType, seed: config.seed)
        return MLShapedArray<Float32>(
            converting: random.normalShapedArray(sampleShape, mean: 0.0, stdev: Double(stdev))
        )
    }

    // MARK: - Embeddings loading

    /// Load pre-computed text embeddings from a binary file
    /// Expected format: raw Float32 tensor, shape [1, seq_len, cap_feat_dim]
    func loadEmbeddings(from url: URL) throws -> MLShapedArray<Float32> {
        let data = try Data(contentsOf: url)
        guard data.count == expectedEmbeddingByteCount else {
            throw Error.invalidEmbeddingsByteCount(actual: data.count, expected: expectedEmbeddingByteCount)
        }
        let floatCount = data.count / MemoryLayout<Float32>.size
        let floats: [Float32] = data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float32.self))
        }
        guard floatCount == expectedEmbeddingFloatCount else {
            throw Error.invalidEmbeddingsByteCount(actual: data.count, expected: expectedEmbeddingByteCount)
        }
        return MLShapedArray<Float32>(scalars: floats, shape: expectedEmbeddingShape)
    }

    // MARK: - Decode

    public func decodeToImages(
        _ latents: [MLShapedArray<Float32>],
        configuration config: ZImageConfiguration
    ) throws -> [CGImage?] {
        // VAE owns its scaling/shift config — no need to pass from pipeline config
        let images = try vae.decode(latents)
        if reduceMemory {
            vae.unloadDecoder()
        }
        return images
    }

    // MARK: - RNG helper

    internal func randomSource(from rng: StableDiffusionRNG, seed: UInt32) -> RandomSource {
        switch rng {
        case .numpyRNG:
            return NumPyRandomSource(seed: seed)
        case .torchRNG:
            return TorchRandomSource(seed: seed)
        case .nvidiaRNG:
            return NvRandomSource(seed: seed)
        }
    }

    private func prepareDebugDirectoryIfNeeded(config: ZImageConfiguration) throws -> URL? {
        guard config.debugEnabled else {
            return nil
        }
        let baseDirectory = config.debugOutputDirectory ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(at: baseDirectory, withIntermediateDirectories: true)
        return baseDirectory
    }

    private func writeDebugTensor(
        _ tensor: MLShapedArray<Float32>,
        kind: String,
        timestep: Int?,
        stepIndex: Int?,
        seed: UInt32,
        outputDirectory: URL?
    ) throws {
        guard let outputDirectory else {
            return
        }

        let filename: String
        if let timestep {
            filename = "\(kind)_t_\(timestep)"
        } else {
            filename = kind
        }

        let binURL = outputDirectory.appending(path: "\(filename).bin")
        let jsonURL = outputDirectory.appending(path: "\(filename).json")

        let scalars = tensor.scalars
        let data = scalars.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
        try data.write(to: binURL, options: .atomic)
        let finiteStats = tensorFiniteStats(tensor)

        let metadata = DebugTensorMetadata(
            kind: kind,
            timestep: timestep,
            stepIndex: stepIndex,
            seed: seed,
            shape: tensor.shape,
            dtype: "float32",
            layout: "row_major",
            scalarCount: tensor.scalarCount,
            byteCount: data.count,
            filename: binURL.lastPathComponent,
            nanCount: finiteStats.nanCount,
            infCount: finiteStats.infCount,
            finiteCount: finiteStats.finiteCount,
            finiteMin: finiteStats.finiteMin,
            finiteMax: finiteStats.finiteMax,
            finiteMean: finiteStats.finiteMean,
            finiteStd: finiteStats.finiteStd
        )
        let metadataData = try JSONEncoder().encode(metadata)
        try metadataData.write(to: jsonURL, options: .atomic)
    }

    private func logFiniteStats(
        _ tensor: MLShapedArray<Float32>,
        label: String,
        timestep: Int?,
        stepIndex: Int?
    ) {
        let stats = tensorFiniteStats(tensor)

        let stepText = stepIndex.map { String($0) } ?? "-"
        let timeText = timestep.map { String($0) } ?? "-"
        let minText = stats.finiteMin.map { String($0) } ?? "nan"
        let maxText = stats.finiteMax.map { String($0) } ?? "nan"
        let meanText = stats.finiteMean.map { String($0) } ?? "nan"
        let stdText = stats.finiteStd.map { String($0) } ?? "nan"

        let message =
            "[ZImageFinite] label=\(label) " +
            "step=\(stepText) " +
            "t=\(timeText) " +
            "nan=\(stats.nanCount) inf=\(stats.infCount) " +
            "finite=\(stats.finiteCount) " +
            "min=\(minText) max=\(maxText) " +
            "mean=\(meanText) std=\(stdText)"

        print(message)
    }

    private func tensorFiniteStats(_ tensor: MLShapedArray<Float32>) -> TensorFiniteStats {
        var nanCount = 0
        var infCount = 0
        var finiteCount = 0
        var finiteMin = Float.greatestFiniteMagnitude
        var finiteMax = -Float.greatestFiniteMagnitude
        var sum: Double = 0
        var sumSquares: Double = 0

        tensor.withUnsafeShapedBufferPointer { buffer, _, _ in
            for value in buffer {
                if value.isNaN {
                    nanCount += 1
                } else if !value.isFinite {
                    infCount += 1
                } else {
                    finiteCount += 1
                    finiteMin = min(finiteMin, value)
                    finiteMax = max(finiteMax, value)
                    let dv = Double(value)
                    sum += dv
                    sumSquares += dv * dv
                }
            }
        }

        let mean: Double?
        let std: Double?
        let minValue: Float?
        let maxValue: Float?
        if finiteCount > 0 {
            mean = sum / Double(finiteCount)
            let variance = max(0.0, (sumSquares / Double(finiteCount)) - (mean! * mean!))
            std = sqrt(variance)
            minValue = finiteMin
            maxValue = finiteMax
        } else {
            mean = nil
            std = nil
            minValue = nil
            maxValue = nil
        }

        return TensorFiniteStats(
            nanCount: nanCount,
            infCount: infCount,
            finiteCount: finiteCount,
            finiteMin: minValue,
            finiteMax: maxValue,
            finiteMean: mean,
            finiteStd: std
        )
    }
}

private struct DebugTensorMetadata: Codable {
    let kind: String
    let timestep: Int?
    let stepIndex: Int?
    let seed: UInt32
    let shape: [Int]
    let dtype: String
    let layout: String
    let scalarCount: Int
    let byteCount: Int
    let filename: String
    let nanCount: Int
    let infCount: Int
    let finiteCount: Int
    let finiteMin: Float?
    let finiteMax: Float?
    let finiteMean: Double?
    let finiteStd: Double?
}

private struct TensorFiniteStats {
    let nanCount: Int
    let infCount: Int
    let finiteCount: Int
    let finiteMin: Float?
    let finiteMax: Float?
    let finiteMean: Double?
    let finiteStd: Double?
}

// MARK: - Progress

@available(iOS 17.0, macOS 14.0, *)
public struct ZImageProgress {
    public let pipeline: ZImagePipelineProtocol
    public let step: Int
    public let stepCount: Int
    public let currentLatentSample: MLShapedArray<Float32>
}
