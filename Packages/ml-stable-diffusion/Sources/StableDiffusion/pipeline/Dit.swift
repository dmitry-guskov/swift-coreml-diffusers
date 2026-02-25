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
            try autoreleasepool {
                try model.loadResources()
                model.unloadResources()
            }
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
        let tNormalized = Float16(Float(1000 - timeStep) / 1000.0)
        let t = MLShapedArray<Float16>(
            scalars: [tNormalized],
            shape: [1]
        )
        
        // Convert hiddenStates (cap_feats) to Float16 for the FP16 CoreML model
        let hiddenStatesF16 = MLShapedArray<Float16>(converting: hiddenStates)
        
        // Build per-sample feature dictionaries with Float16 inputs
        let inputs: [MLDictionaryFeatureProvider] = try latents.map { latent in
            // Convert latent to Float16 at the model boundary
            let latentF16 = MLShapedArray<Float16>(converting: latent)
            let dict: [String: Any] = [
                "latents": MLMultiArray(latentF16),
                "timestep": MLMultiArray(t),
                "cap_feats": MLMultiArray(hiddenStatesF16)
            ]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)
        print("[Dit] predictNoise: timeStep=\(timeStep) latent_shape=\(latents.first?.shape ?? []) cap_feats_shape=\(hiddenStates.shape) batch_count=\(batch.count)")
        let results = try predictions(from: batch)
        // Extract results and convert FP16 output back to Float32
        // Match Python sign convention: noise_pred = -model_out.squeeze(2)
        let noise: [MLShapedArray<Float32>] = (0..<results.count).map { i in
            let result = results.features(at: i)
            let outputName = result.featureNames.first!
            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!
            // Model outputs Float16; convert to Float32 for scheduler math
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
        var results: MLBatchProvider
        do {
            results = try models.first!.perform { model in
                print("[Dit] Starting predictions(fromBatch:) on model...")
                let r = try model.predictions(fromBatch: batch)
                print("[Dit] predictions(fromBatch:) succeeded")
                return r
            }
        } catch {
            print("[Dit] PREDICTION FAILED: \(error)")
            print("[Dit] Error domain: \(String(describing: (error as NSError).domain))")
            print("[Dit] Error code: \((error as NSError).code)")
            print("[Dit] Error userInfo: \((error as NSError).userInfo)")
            throw error
        }
        if models.count == 1 {
            return results
        }
        let inputs = batch.arrayOfFeatureValueDictionaries
        for (i, stage) in models.dropFirst().enumerated() {
            let next = try results.arrayOfFeatureValueDictionaries
                .enumerated().map { (index, dict) in
                    let merged = dict.merging(inputs[index]) { output, _ in output }
                    return try MLDictionaryFeatureProvider(dictionary: merged)
                }
            let nextBatch = MLArrayBatchProvider(array: next)
            do {
                results = try stage.perform { model in
                    print("[Dit] Starting predictions for chunk \(i + 1)...")
                    let r = try model.predictions(fromBatch: nextBatch)
                    print("[Dit] Chunk \(i + 1) predictions succeeded")
                    return r
                }
            } catch {
                print("[Dit] CHUNK \(i + 1) PREDICTION FAILED: \(error)")
                print("[Dit] Error: \((error as NSError).domain) code=\((error as NSError).code)")
                throw error
            }
        }
        return results
    }
}
