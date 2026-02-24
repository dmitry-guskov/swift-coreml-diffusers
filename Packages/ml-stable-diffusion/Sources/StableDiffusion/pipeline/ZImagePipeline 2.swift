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
        if reduceMemory {
            try prewarmResources()
        } else {
            try dit.loadResources()
            try vae.loadResources()
        }
    }

    public func unloadResources() {
        dit.unloadResources()
        vae.unloadResources()
    }

    public func prewarmResources() throws {
        try dit.prewarmResources()
        try vae.prewarmResources()
    }

    // MARK: - Image generation

    public func generateImages(
        configuration config: ZImageConfiguration,
        progressHandler: (ZImageProgress) -> Bool = { _ in true }
    ) throws -> [CGImage?] {

        // Load pre-computed text embeddings from file
        let hiddenStates = try loadEmbeddings(from: config.embeddingsURL)

        // Setup scheduler (single instance — one image)
        let scheduler: Scheduler = DiscreteFlowScheduler(
            stepCount: config.stepCount,
            timeStepShift: config.schedulerTimestepShift
        )

        // Generate random initial latent noise
        var latent = try generateLatentSample(configuration: config, scheduler: scheduler)

        // Will hold the denoised intermediate for the decoder
        var denoisedLatent = latent

        // De-noising loop
        let timeSteps: [Int] = scheduler.calculateTimesteps(strength: nil)
        for (step, t) in timeSteps.enumerated() {

            // Predict noise residual conditioned on text embeddings
            let noise = try dit.predictNoise(
                latents: [latent],
                timeStep: t,
                hiddenStates: hiddenStates
            )

            // Scheduler step: compute previous latent sample
            latent = scheduler.step(
                output: noise[0],
                timeStep: t,
                sample: latent
            )

            denoisedLatent = scheduler.modelOutputs.last ?? latent

            let currentSample = config.useDenoisedIntermediates ? denoisedLatent : latent

            // Report progress
            let progress = ZImageProgress(
                pipeline: self,
                step: step,
                stepCount: timeSteps.count,
                currentLatentSample: currentSample
            )
            if !progressHandler(progress) {
                return []
            }
        }

        if reduceMemory {
            dit.unloadResources()
        }

        // Decode the final latent to an image
        return try decodeToImages([denoisedLatent], configuration: config)
    }

    // MARK: - Latent generation

    func generateLatentSample(
        configuration config: ZImageConfiguration,
        scheduler: Scheduler
    ) throws -> MLShapedArray<Float32> {
        var sampleShape = dit.latentSampleShape
        sampleShape[0] = 1

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
        let floatCount = data.count / MemoryLayout<Float32>.size
        let floats: [Float32] = data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float32.self))
        }
        // Shape must match what the DiT model expects for encoder_hidden_states
        // Adjust this shape based on your actual embeddings file format
        // For example: [1, seq_len, 2560] where 2560 = cap_feat_dim
        // TODO: read shape from a sidecar or embed in the file header
        return MLShapedArray<Float32>(scalars: floats, shape: [1, floats.count / 2560, 2560])
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
}

// MARK: - Progress

@available(iOS 17.0, macOS 14.0, *)
public struct ZImageProgress {
    public let pipeline: ZImagePipelineProtocol
    public let step: Int
    public let stepCount: Int
    public let currentLatentSample: MLShapedArray<Float32>
}