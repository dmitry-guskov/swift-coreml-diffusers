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
/// Supports stagewise execution where each stage has a different I/O contract:
///   - stage 0:  inputs `latents`, `timestep`, `cap_feats`  →  output `hidden_tokens`
///   - stage 1‒4: inputs `hidden_tokens`, `timestep`, `cap_feats`  →  output `hidden_tokens`
///   - stage 5:  inputs `hidden_tokens`, `timestep`, `cap_feats`  →  output final noise tensor
@available(iOS 17.0, macOS 14.0, *)
public struct Dit: ResourceManaging {

    public enum Error: Swift.Error, LocalizedError {
        case invalidStageInputs(stageIndex: Int, expected: Set<String>, actual: [String])
        case invalidOutputContract(stageIndex: Int, outputKeys: [String])
        case missingOutputFeature(stageIndex: Int)
        case incompatibleHiddenTokens(stageIndex: Int, expected: [Int], actual: [Int])
        case incompatibleHiddenTokensDtype(stageIndex: Int, expected: MLMultiArrayDataType, actual: MLMultiArrayDataType)

        public var errorDescription: String? {
            switch self {
            case .invalidStageInputs(let idx, let expected, let actual):
                return "Stage \(idx) input contract mismatch. Expected \(expected.sorted()), got \(actual)."
            case .invalidOutputContract(let idx, let outputKeys):
                return "Stage \(idx) must expose exactly one output tensor; got \(outputKeys)."
            case .missingOutputFeature(let idx):
                return "Stage \(idx) output feature is missing from prediction result."
            case .incompatibleHiddenTokens(let idx, let expected, let actual):
                return "Stage \(idx) hidden_tokens shape mismatch. Expected \(expected), got \(actual)."
            case .incompatibleHiddenTokensDtype(let idx, let expected, let actual):
                return "Stage \(idx) hidden_tokens dtype mismatch. Expected \(expected), got \(actual)."
            }
        }
    }

    private static let stage0Inputs: Set<String> = ["latents", "timestep", "cap_feats"]
    private static let stageNInputs: Set<String> = ["hidden_tokens", "timestep", "cap_feats"]

    // MARK: - Properties

    var models: [ManagedMLModel]
    let tScale: Float

    // MARK: - Initializers

    public init(modelAt url: URL,
                configuration: MLModelConfiguration,
                tScale: Float = 1000.0) {
        self.models = [ManagedMLModel(modelAt: url, configuration: configuration)]
        self.tScale = tScale
    }

    public init(stagesAt urls: [URL],
                configuration: MLModelConfiguration,
                tScale: Float = 1000.0) {
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.tScale = tScale
    }

    var unloadStagesAfterUse: Bool { models.count > 1 }

    // MARK: - ResourceManaging

    public func loadResources() throws {
        if unloadStagesAfterUse {
            try prewarmResources()
        } else {
            for (i, model) in models.enumerated() {
                try model.loadResources()
                try model.perform { loaded in
                    try validateStageInputContract(for: loaded, stageIndex: i)
                }
            }
        }
    }

    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }

    public func prewarmResources() throws {
        for (i, model) in models.enumerated() {
            try autoreleasepool {
                try model.loadResources()
                try model.perform { loaded in
                    try validateStageInputContract(for: loaded, stageIndex: i)
                    try validateSingleOutput(for: loaded, stageIndex: i)
                }
                model.unloadResources()
            }
        }
    }

    // MARK: - Validation

    private func expectedInputKeys(for stageIndex: Int) -> Set<String> {
        stageIndex == 0 ? Self.stage0Inputs : Self.stageNInputs
    }

    private func validateStageInputContract(for model: MLModel, stageIndex: Int) throws {
        let expected = expectedInputKeys(for: stageIndex)
        let actual = Set(model.modelDescription.inputDescriptionsByName.keys)
        guard actual == expected else {
            throw Error.invalidStageInputs(stageIndex: stageIndex, expected: expected, actual: actual.sorted())
        }
    }

    private func validateSingleOutput(for model: MLModel, stageIndex: Int) throws {
        let outputs = model.modelDescription.outputDescriptionsByName.keys.sorted()
        guard outputs.count == 1 else {
            throw Error.invalidOutputContract(stageIndex: stageIndex, outputKeys: outputs)
        }
    }

    private func singleOutputTensor(from provider: MLFeatureProvider, stageIndex: Int) throws -> MLMultiArray {
        let names = provider.featureNames.sorted()
        guard names.count == 1,
              let name = names.first,
              let value = provider.featureValue(for: name)?.multiArrayValue else {
            throw Error.missingOutputFeature(stageIndex: stageIndex)
        }
        return value
    }

    private func validateHiddenTokensContinuity(
        previousOutput: MLMultiArray,
        nextModel: MLModel,
        stageIndex: Int
    ) throws {
        guard let desc = nextModel.modelDescription.inputDescriptionsByName["hidden_tokens"],
              let constraint = desc.multiArrayConstraint else { return }
        let expectedShape = constraint.shape.map(\.intValue)
        let actualShape = previousOutput.shape.map(\.intValue)
        if expectedShape != actualShape {
            throw Error.incompatibleHiddenTokens(stageIndex: stageIndex, expected: expectedShape, actual: actualShape)
        }
        if constraint.dataType != previousOutput.dataType {
            throw Error.incompatibleHiddenTokensDtype(stageIndex: stageIndex, expected: constraint.dataType, actual: previousOutput.dataType)
        }
    }

    // MARK: - Model metadata helpers

    var latentSampleDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["latents"]!
        }
    }

    public var latentSampleShape: [Int] {
        latentSampleDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    // MARK: - Noise prediction

    func predictNoise(
        latents: [MLShapedArray<Float32>],
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>
    ) throws -> [MLShapedArray<Float32>] {
        let tNormalized = Float16(Float(1000 - timeStep) / 1000.0)
        let t = MLShapedArray<Float16>(scalars: [tNormalized], shape: [1])
        let hiddenStatesF16 = MLShapedArray<Float16>(converting: hiddenStates)

        let inputs: [MLDictionaryFeatureProvider] = try latents.map { latent in
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
        var noise: [MLShapedArray<Float32>] = []
        noise.reserveCapacity(results.count)
        for i in 0..<results.count {
            let result = results.features(at: i)
            let outputArray = try singleOutputTensor(from: result, stageIndex: models.count - 1)
            let fp32 = MLMultiArray(concatenating: [outputArray], axis: 0, dataType: .float32)
            let modelOut = MLShapedArray<Float32>(fp32)
            noise.append(
                MLShapedArray<Float32>(
                    scalars: modelOut.scalars.map { -$0 },
                    shape: modelOut.shape
                )
            )
        }
        return noise
    }

    // MARK: - Stagewise prediction

    /// Runs predictions through all stages sequentially, piping outputs forward.
    ///
    /// Stage 0 receives the original batch (`latents`, `timestep`, `cap_feats`).
    /// Its single output tensor is mapped to `hidden_tokens` and merged with
    /// `timestep` and `cap_feats` from the original batch for stages 1‒5.
    /// Each stage is unloaded immediately after use when running multi-stage.
    func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {
        let shouldUnload = unloadStagesAfterUse

        let originalInputs = batch.arrayOfFeatureValueDictionaries
        var accumulated: [[String: MLFeatureValue]] = originalInputs
        var lastResults: MLBatchProvider!

        for (stageIndex, stage) in models.enumerated() {
            do {
                lastResults = try autoreleasepool {
                    let inputBatch = MLArrayBatchProvider(array: try accumulated.map {
                        try MLDictionaryFeatureProvider(dictionary: $0)
                    })
                    let r = try stage.perform { model in
                        try validateStageInputContract(for: model, stageIndex: stageIndex)
                        try validateSingleOutput(for: model, stageIndex: stageIndex)

                        if stageIndex > 0, let htDesc = model.modelDescription.inputDescriptionsByName["hidden_tokens"] {
                            for entry in accumulated {
                                if let ht = entry["hidden_tokens"]?.multiArrayValue, let constraint = htDesc.multiArrayConstraint {
                                    let expectedShape = constraint.shape.map(\.intValue)
                                    let actualShape = ht.shape.map(\.intValue)
                                    if expectedShape != actualShape {
                                        throw Error.incompatibleHiddenTokens(stageIndex: stageIndex, expected: expectedShape, actual: actualShape)
                                    }
                                    if constraint.dataType != ht.dataType {
                                        throw Error.incompatibleHiddenTokensDtype(stageIndex: stageIndex, expected: constraint.dataType, actual: ht.dataType)
                                    }
                                }
                            }
                        }

                        print("[Dit] Running stage \(stageIndex)...")
                        let inferenceStart = CFAbsoluteTimeGetCurrent()
                        let r = try model.predictions(fromBatch: inputBatch)
                        let inferenceElapsed = CFAbsoluteTimeGetCurrent() - inferenceStart
                        print("[Dit] Stage \(stageIndex) succeeded in \(String(format: "%.2f", inferenceElapsed))s")
                        return r
                    }
                    if shouldUnload {
                        print("[Dit] Unloading stage \(stageIndex)")
                        stage.unloadResources()
                    }
                    return r
                }

                let outputs = lastResults.arrayOfFeatureValueDictionaries
                accumulated = try accumulated.enumerated().map { (batchIdx, prev) in
                    let outputDict = outputs[batchIdx]
                    guard let outputKey = outputDict.keys.first,
                          let outputValue = outputDict[outputKey] else {
                        throw Error.missingOutputFeature(stageIndex: stageIndex)
                    }

                    var next = prev
                    next["hidden_tokens"] = outputValue
                    next.removeValue(forKey: "latents")
                    return next
                }
            } catch {
                print("[Dit] STAGE \(stageIndex) FAILED: \(error)")
                print("[Dit] Error: \((error as NSError).domain) code=\((error as NSError).code)")
                throw error
            }
        }
        return lastResults
    }
}

// MARK: - MLBatchProvider convenience

extension MLBatchProvider {
    var arrayOfFeatureValueDictionaries: [[String: MLFeatureValue]] {
        (0..<self.count).map {
            self.features(at: $0).featureValueDictionary
        }
    }
}

extension MLFeatureProvider {
    var featureValueDictionary: [String: MLFeatureValue] {
        var dict: [String: MLFeatureValue] = [:]
        for name in self.featureNames {
            if let value = self.featureValue(for: name) {
                dict[name] = value
            }
        }
        return dict
    }
}
