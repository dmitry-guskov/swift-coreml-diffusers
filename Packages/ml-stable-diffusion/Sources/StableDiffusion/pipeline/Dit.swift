//
//  Dit.swift
//  stable-diffusion
//
//  Created by Dmitry Guskov on 10.02.2026.
//

import Foundation
import CoreML

/// Context passed from the pipeline to enable per-stage tensor dumps.
@available(iOS 18.0, macOS 14.0, *)
public struct DitDebugContext {
    public let stagesDirectory: URL
    public let stepIndex: Int
    public let timestep: Int
    public let seed: UInt32

    public init(stagesDirectory: URL, stepIndex: Int, timestep: Int, seed: UInt32) {
        self.stagesDirectory = stagesDirectory
        self.stepIndex = stepIndex
        self.timestep = timestep
        self.seed = seed
    }
}

/// DiT (Diffusion Transformer) noise prediction model for Z-Image
///
/// Supports stagewise execution with a split preamble:
///   - stage 0 (image preamble):   inputs `latents`, `timestep`  →  output `image_tokens`
///   - stage 1 (caption preamble): inputs `cap_feats`            →  output `cap_tokens`
///   - Swift concatenates image_tokens + cap_tokens along axis 1  →  `hidden_tokens`
///   - stages 2‒N: inputs `hidden_tokens`, `timestep` [+ `lora_vec`, `lora_scale`]
@available(iOS 18.0, macOS 14.0, *)
public struct Dit: ResourceManaging {

    public enum Error: Swift.Error, LocalizedError {
        case invalidStageInputs(stageIndex: Int, expectedVariants: [[String]], actual: [String])
        case invalidOutputContract(stageIndex: Int, outputKeys: [String])
        case missingOutputFeature(stageIndex: Int)
        case incompatibleHiddenTokens(stageIndex: Int, expected: [Int], actual: [Int])
        case incompatibleHiddenTokensDtype(stageIndex: Int, expected: MLMultiArrayDataType, actual: MLMultiArrayDataType)
        case missingImageTokensForConcatenation

        public var errorDescription: String? {
            switch self {
            case .invalidStageInputs(let idx, let expectedVariants, let actual):
                return "Stage \(idx) input contract mismatch. Expected one of \(expectedVariants), got \(actual)."
            case .invalidOutputContract(let idx, let outputKeys):
                return "Stage \(idx) must expose exactly one output tensor; got \(outputKeys)."
            case .missingOutputFeature(let idx):
                return "Stage \(idx) output feature is missing from prediction result."
            case .incompatibleHiddenTokens(let idx, let expected, let actual):
                return "Stage \(idx) hidden_tokens shape mismatch. Expected \(expected), got \(actual)."
            case .incompatibleHiddenTokensDtype(let idx, let expected, let actual):
                return "Stage \(idx) hidden_tokens dtype mismatch. Expected \(expected), got \(actual)."
            case .missingImageTokensForConcatenation:
                return "image_tokens from stage 0 not available when concatenating after stage 1."
            }
        }
    }

    private static let loraInputs: Set<String> = ["lora_vec", "lora_scale"]
    private static let stage0Inputs: Set<String> = ["latents", "timestep"]
    private static let stage1Inputs: Set<String> = ["cap_feats"]
    private static let stageNInputs: Set<String> = ["hidden_tokens", "timestep"]
    private static let legacyStage0Inputs: Set<String> = ["latents", "timestep", "cap_feats"]

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

    private func expectedInputKeyVariants(for stageIndex: Int) -> [Set<String>] {
        if models.count == 1 {
            let base = Self.legacyStage0Inputs
            return [base, base.union(Self.loraInputs)]
        }
        switch stageIndex {
        case 0:
            return [Self.stage0Inputs]
        case 1:
            return [Self.stage1Inputs]
        default:
            return [Self.stageNInputs, Self.stageNInputs.union(Self.loraInputs)]
        }
    }

    private func modelExpectsLoRAInputs(_ model: MLModel) -> Bool {
        let actual = Set(model.modelDescription.inputDescriptionsByName.keys)
        return Self.loraInputs.isSubset(of: actual)
    }

    private func validateStageInputContract(for model: MLModel, stageIndex: Int) throws {
        let expectedVariants = expectedInputKeyVariants(for: stageIndex)
        let actual = Set(model.modelDescription.inputDescriptionsByName.keys)
        guard expectedVariants.contains(actual) else {
            let rendered = expectedVariants.map { Array($0).sorted() }
            throw Error.invalidStageInputs(stageIndex: stageIndex, expectedVariants: rendered, actual: actual.sorted())
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
        hiddenStates: MLShapedArray<Float32>,
        loraInputProvider: ZImageLoRAInputProvider? = nil,
        debugContext: DitDebugContext? = nil
    ) throws -> [MLShapedArray<Float32>] {
        let tNormalized = Float32(1000 - timeStep) / 1000.0
        let t = MLShapedArray<Float32>(scalars: [tNormalized], shape: [1])

        let inputs: [MLDictionaryFeatureProvider] = try latents.map { latent in
            let dict: [String: Any] = [
                "latents": MLMultiArray(latent),
                "timestep": MLMultiArray(t),
                "cap_feats": MLMultiArray(hiddenStates)
            ]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)
        print("[Dit] predictNoise: timeStep=\(timeStep) latent_shape=\(latents.first?.shape ?? []) cap_feats_shape=\(hiddenStates.shape) batch_count=\(batch.count)")
        let results = try predictions(from: batch, loraInputProvider: loraInputProvider, debugContext: debugContext)
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
    /// Split preamble (multi-stage, ≥ 12 stages):
    ///   - Stage 0 (image preamble): receives `{latents, timestep}` → `image_tokens`
    ///   - Stage 1 (caption preamble): receives `{cap_feats}` → `cap_tokens`
    ///   - Swift concatenates `image_tokens + cap_tokens` along axis 1 → `hidden_tokens`
    ///   - Stages 2‒N: receive `{hidden_tokens, timestep}` [+ LoRA] → `hidden_tokens`
    ///
    /// Legacy single-model mode passes the full input dict unchanged.
    func predictions(
        from batch: MLBatchProvider,
        loraInputProvider: ZImageLoRAInputProvider? = nil,
        debugContext: DitDebugContext? = nil
    ) throws -> MLBatchProvider {
        let shouldUnload = unloadStagesAfterUse
        let isMultiStage = models.count > 1

        let originalInputs = batch.arrayOfFeatureValueDictionaries
        var accumulated: [[String: MLFeatureValue]] = originalInputs
        var lastResults: MLBatchProvider!
        var zeroLoRAInputProvider: ZImageLoRAInputProvider?

        for (stageIndex, stage) in models.enumerated() {
            do {
                lastResults = try autoreleasepool {
                    let r = try stage.perform { model in
                        try validateStageInputContract(for: model, stageIndex: stageIndex)
                        try validateSingleOutput(for: model, stageIndex: stageIndex)

                        let inputBatch = MLArrayBatchProvider(array: try accumulated.map { entry in
                            var prepared: [String: MLFeatureValue]

                            if isMultiStage && stageIndex == 0 {
                                prepared = entry.filter { Self.stage0Inputs.contains($0.key) }
                            } else if isMultiStage && stageIndex == 1 {
                                prepared = entry.filter { Self.stage1Inputs.contains($0.key) }
                            } else {
                                prepared = entry
                                if modelExpectsLoRAInputs(model) {
                                    let provider: ZImageLoRAInputProvider
                                    if let loraInputProvider {
                                        provider = loraInputProvider
                                    } else if let zeroLoRAInputProvider {
                                        provider = zeroLoRAInputProvider
                                    } else {
                                        let created = try ZImageLoRAInputProvider(loraURL: nil, scale: 0)
                                        zeroLoRAInputProvider = created
                                        provider = created
                                    }
                                    let features = try provider.featureValues(for: stageIndex)
                                    prepared["lora_vec"] = features.vector
                                    prepared["lora_scale"] = features.scale
                                }
                            }

                            return try MLDictionaryFeatureProvider(dictionary: prepared)
                        })

                        if stageIndex >= 2, let htDesc = model.modelDescription.inputDescriptionsByName["hidden_tokens"] {
                            for entry in accumulated {
                                if let ht = entry["hidden_tokens"]?.multiArrayValue, let constraint = htDesc.multiArrayConstraint {
                                    let expectedShape = constraint.shape.map(\.intValue)
                                    let actualShape = ht.shape.map(\.intValue)
                                    print("[Dit] Stage \(stageIndex) input hidden_tokens: dtype=\(ht.dataType.rawValue) shape=\(actualShape) expected_dtype=\(constraint.dataType.rawValue) expected_shape=\(expectedShape)")
                                    if expectedShape != actualShape {
                                        throw Error.incompatibleHiddenTokens(stageIndex: stageIndex, expected: expectedShape, actual: actualShape)
                                    }
                                    if constraint.dataType != ht.dataType {
                                        print("[Dit] Stage \(stageIndex) hidden_tokens dtype mismatch: model expects \(constraint.dataType.rawValue), got \(ht.dataType.rawValue) — CoreML will auto-cast")
                                    }
                                }
                            }
                        }

                        for (key, fv) in accumulated.first ?? [:] {
                            if let ma = fv.multiArrayValue {
                                print("[Dit] Stage \(stageIndex) input '\(key)': dtype=\(ma.dataType.rawValue) shape=\(ma.shape.map(\.intValue))")
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

                    if let ma = outputValue.multiArrayValue {
                        let shape = ma.shape.map(\.intValue)
                        print("[Dit] Stage \(stageIndex) output: key='\(outputKey)' dtype=\(ma.dataType.rawValue) shape=\(shape)")
                        let statsArray = (ma.dataType == .float32) ? ma : MLMultiArray(concatenating: [ma], axis: 0, dataType: .float32)
                        var finiteCount = 0, nanCount = 0, infCount = 0
                        let ptr = statsArray.dataPointer.bindMemory(to: Float32.self, capacity: statsArray.count)
                        for idx in 0..<statsArray.count {
                            let v = ptr[idx]
                            if v.isNaN { nanCount += 1 }
                            else if !v.isFinite { infCount += 1 }
                            else { finiteCount += 1 }
                        }
                        print("[Dit] Stage \(stageIndex) output stats: finite=\(finiteCount) nan=\(nanCount) inf=\(infCount) total=\(ma.count)")

                        if let ctx = debugContext {
                            try Self.writeStageOutput(ma, stageIndex: stageIndex, context: ctx)
                        }
                    }

                    var next = prev

                    if isMultiStage && stageIndex == 0 {
                        next["image_tokens"] = outputValue
                        next.removeValue(forKey: "latents")
                    } else if isMultiStage && stageIndex == 1 {
                        guard let imageTokens = next["image_tokens"]?.multiArrayValue,
                              let capTokens = outputValue.multiArrayValue else {
                            throw Error.missingImageTokensForConcatenation
                        }
                        let hidden = MLMultiArray(concatenating: [imageTokens, capTokens], axis: 1, dataType: imageTokens.dataType)
                        print("[Dit] Concatenated image_tokens + cap_tokens → hidden_tokens: shape=\(hidden.shape.map(\.intValue)) dtype=\(hidden.dataType.rawValue)")
                        next["hidden_tokens"] = MLFeatureValue(multiArray: hidden)
                        next.removeValue(forKey: "image_tokens")
                        next.removeValue(forKey: "cap_feats")
                    } else {
                        next["hidden_tokens"] = outputValue
                        next.removeValue(forKey: "latents")
                        next.removeValue(forKey: "cap_feats")
                    }

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

    // MARK: - Stage debug output

    private static func writeStageOutput(
        _ ma: MLMultiArray,
        stageIndex: Int,
        context ctx: DitDebugContext
    ) throws {
        let stem = "step\(ctx.stepIndex)_t\(ctx.timestep)_stage\(stageIndex)"
        let binURL = ctx.stagesDirectory.appending(path: "\(stem).bin")
        let jsonURL = ctx.stagesDirectory.appending(path: "\(stem).json")

        let count = ma.count
        let shape = ma.shape.map(\.intValue)
        let fp32 = MLMultiArray(concatenating: [ma], axis: 0, dataType: .float32)
        let ptr = fp32.dataPointer.bindMemory(to: Float32.self, capacity: count)

        var nanCount = 0, infCount = 0, finiteCount = 0
        var fMin = Float.greatestFiniteMagnitude
        var fMax = -Float.greatestFiniteMagnitude
        var sum: Double = 0, sumSq: Double = 0

        let raw = UnsafeMutableBufferPointer(start: UnsafeMutablePointer(mutating: ptr), count: count)
        for v in raw {
            if v.isNaN { nanCount += 1 }
            else if !v.isFinite { infCount += 1 }
            else {
                finiteCount += 1
                fMin = min(fMin, v); fMax = max(fMax, v)
                let d = Double(v); sum += d; sumSq += d * d
            }
        }

        let data = Data(bytes: ptr, count: count * MemoryLayout<Float32>.size)
        try data.write(to: binURL, options: .atomic)

        let mean = finiteCount > 0 ? sum / Double(finiteCount) : 0
        let variance = finiteCount > 0 ? max(0, sumSq / Double(finiteCount) - mean * mean) : 0

        let meta: [String: Any] = [
            "kind": "stage_output",
            "stage_index": stageIndex,
            "step_index": ctx.stepIndex,
            "timestep": ctx.timestep,
            "seed": ctx.seed,
            "shape": shape,
            "dtype": "float32",
            "layout": "row_major",
            "scalar_count": count,
            "byte_count": data.count,
            "filename": binURL.lastPathComponent,
            "nan_count": nanCount,
            "inf_count": infCount,
            "finite_count": finiteCount,
            "finite_min": finiteCount > 0 ? fMin : NSNull(),
            "finite_max": finiteCount > 0 ? fMax : NSNull(),
            "finite_mean": finiteCount > 0 ? mean : NSNull(),
            "finite_std": finiteCount > 0 ? sqrt(variance) : NSNull(),
        ]
        let jsonData = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: jsonURL, options: .atomic)

        print("[Dit] Saved stage \(stageIndex) output → \(binURL.lastPathComponent) (\(count) floats, shape=\(shape))")
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
