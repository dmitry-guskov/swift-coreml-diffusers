//
//  Dit.swift
//  stable-diffusion
//
//  Created by Dmitry Guskov on 10.02.2026.
//

import Foundation
import CoreML

/// DiT (Diffusion Transformer) noise prediction model for Z-Image
///
/// Wraps the CoreML-exported ZImageTransformer2DModel. The Python model
/// performs patchification, RoPE, adaLN modulation, and attention internally;
/// after CoreML conversion all of that is baked into the compiled model so
/// the Swift side only needs to supply:
///   - `latents`                — latent tensor  [1, C, H, W]  (F dim squeezed at export)
///   - `timestep`               — scalar or [1] float timestep
///   - `cap_feats`              — caption features [1, seq_len, cap_feat_dim]
@available(iOS 17.0, macOS 14.0, *)
public struct Dit: ResourceManaging {
    public enum Error: Swift.Error, LocalizedError {
        case inputContractMismatch(expected: [String], actual: [String])

        public var errorDescription: String? {
            switch self {
            case .inputContractMismatch(let expected, let actual):
                return "DiT CoreML input contract mismatch. Expected \(expected), got \(actual)."
            }
        }
    }
    
    // MARK: - Properties
    
    /// Underlying Core ML model(s). May be chunked for memory efficiency.
    var models: [ManagedMLModel]
    
    /// Timestep scale matching Python `ZImageTransformer2DModel.t_scale`
    /// The Python pipeline sends `(1000 - t) / 1000` which the model
    /// multiplies back by 1000 internally.  If your CoreML export already
    /// embeds that multiplication set this to 1.0.
    let tScale: Float
    
    // MARK: - Initializers
    
    /// Creates a DiT model from a single compiled Core ML model
    ///
    /// - Parameters:
    ///   - url: Location of the compiled `.mlmodelc`
    ///   - configuration: Core ML configuration (compute units, etc.)
    ///   - tScale: Timestep scaling factor (default 1000, matching Python)
    public init(modelAt url: URL,
                configuration: MLModelConfiguration,
                tScale: Float = 1000.0) {
        self.models = [ManagedMLModel(modelAt: url, configuration: configuration)]
        self.tScale = tScale
    }
    
    /// Creates a DiT model from multiple compiled chunks
    ///
    /// - Parameters:
    ///   - urls: Ordered URLs to each compiled chunk
    ///   - configuration: Core ML configuration
    ///   - tScale: Timestep scaling factor
    public init(chunksAt urls: [URL],
                configuration: MLModelConfiguration,
                tScale: Float = 1000.0) {
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.tScale = tScale
    }
    
    // MARK: - ResourceManaging
    
    public func loadResources() throws {
        for model in models {
            try model.loadResources()
            try model.perform { loadedModel in
                try validateInputContract(for: loadedModel)
            }
        }
    }
    
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }
    
    public func prewarmResources() throws {
        for model in models {
            try model.loadResources()
            model.unloadResources()
        }
    }
    
    // MARK: - Model metadata helpers
    
    var latentSampleDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["latents"]!
        }
    }

    private func validateInputContract(for model: MLModel) throws {
        let expected = ["cap_feats", "latents", "timestep"]
        let actual = model.modelDescription.inputDescriptionsByName.keys.sorted()
        guard expected == actual else {
            throw Error.inputContractMismatch(expected: expected, actual: actual)
        }
    }
    
    /// The expected shape of the latent sample input (e.g. [1, 16, H, W])
    public var latentSampleShape: [Int] {
        latentSampleDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
    
    // MARK: - Noise prediction
    
    /// Predict noise residuals from latent samples
    ///
    /// Mirrors the Python pipeline's call:
    ///
    /// timestep = (1000 - t) / 1000          # normalize
    /// model_out = transformer(latents, timestep, cap_feats)
    /// noise_pred = -model_out.squeeze(2)     # remove frame dim
    ///
    ///
    /// - Parameters:
    /// - latents: Batch of latent samples [1, C, H, W]
    /// - timeStep: Current diffusion timestep (integer from scheduler, e.g. 0…1000)
    /// - hiddenStates: Caption / text encoder hidden states [1, seq_len, dim]
    /// - Returns: Array of predicted noise residuals [1, C, H, W]
    func predictNoise(
        latents: [MLShapedArray<Float32>],
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>
    ) throws -> [MLShapedArray<Float32>] {
        // Normalize timestep the same way the Python pipeline does:
        // timestep = (1000 - t) / 1000
        // The CoreML model internally multiplies by t_scale if that was
        // preserved during export. Adjust if your export differs.
        let tNormalized = Float(1000 - timeStep) / 1000.0
        let t = MLShapedArray<Float32>(
            scalars: [tNormalized],
            shape: [1]
        )
        // Build per-sample feature dictionaries
        let inputs: [MLDictionaryFeatureProvider] = try latents.map { latent in
            let dict: [String: Any] = [
                "latents": MLMultiArray(latent),
                "timestep": MLMultiArray(t),
                "cap_feats": MLMultiArray(hiddenStates)
            ]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)
        // Run through model (possibly multi-stage / chunked)
        let results = try predictions(from: batch)
        // Extract results as Float32 and match Python sign convention:
        // noise_pred = -model_out.squeeze(2)
        let noise: [MLShapedArray<Float32>] = (0..<results.count).map { i in
            let result = results.features(at: i)
            let outputName = result.featureNames.first!
            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!
            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            let modelOut = MLShapedArray<Float32>(fp32Noise)
            return MLShapedArray<Float32>(
                scalars: modelOut.scalars.map { -$0 },
                shape: modelOut.shape
            )
        }
        return noise
    }
    // MARK: - Multi-stage prediction (chunked model support)
    /// Runs predictions through all model stages, piping outputs forward.
    /// Identical in structure to Unet.predictions(from:).
    func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {
        var results = try models.first!.perform { model in
            try model.predictions(fromBatch: batch)
        }
        if models.count == 1 {
            return results
        }
        // Manual pipeline: feed previous outputs + original inputs to next stage
        let inputs = batch.arrayOfFeatureValueDictionaries
        for stage in models.dropFirst() {
            let next = try results.arrayOfFeatureValueDictionaries
                .enumerated().map { (index, dict) in
                    let merged = dict.merging(inputs[index]) { output, _ in output }
                    return try MLDictionaryFeatureProvider(dictionary: merged)
                }
            let nextBatch = MLArrayBatchProvider(array: next)
            results = try stage.perform { model in
                try model.predictions(fromBatch: nextBatch)
            }
        }
        return results
    }
}
