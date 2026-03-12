//
//  State.swift
//  Diffusion
//
//  Created by Pedro Cuenca on 17/1/23.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import Combine
import SwiftUI
import StableDiffusion
import CoreML

// MARK: - Types formerly in Pipeline.swift (deleted with old SD code)

struct StableDiffusionProgress {
    var step: Int
    var stepCount: Int
    var currentImages: [CGImage?] = []
}

struct GenerationResult {
    var image: CGImage?
    var lastSeed: UInt32
    var interval: TimeInterval?
    var userCanceled: Bool
    var itsPerSecond: Double?
    var initialNoiseData: Data?
    var initialNoiseShape: [Int]?
}

protocol AppPipeline: AnyObject {
    var progressPublisher: CurrentValueSubject<StableDiffusionProgress?, Never> { get }

    func generate(
        prompt: String,
        negativePrompt: String,
        scheduler: StableDiffusionScheduler,
        numInferenceSteps: Int,
        seed: UInt32,
        numPreviews: Int,
        guidanceScale: Float,
        disableSafety: Bool,
        startingImage: CGImage?,
        strength: Float?,
        initialNoiseData: Data?,
        initialNoiseShape: [Int]?,
        interpolationBaseNoiseData: Data?,
        interpolationBaseNoiseShape: [Int]?,
        interpolationSeed: UInt32?,
        interpolationAmount: Float?
    ) throws -> GenerationResult

    func setCancelled()
}

let DEFAULT_MODEL = ModelInfo.sd3
let DEFAULT_PROMPT = "Labrador in the style of Vermeer"

enum GenerationState {
    case startup
    case running(StableDiffusionProgress?)
    case complete(String, CGImage?, UInt32, TimeInterval?)
    case userCanceled
    case failed(Error)
}

typealias ComputeUnits = MLComputeUnits

private struct InitialLatentFileError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct GenerationProgressSnapshot {
    let step: Int
    let stepCount: Int
    let fraction: Double
    let iterationsPerSecond: Double?
    let etaSeconds: Double?
    let elapsedSeconds: Double
    let phaseText: String

    static let idle = GenerationProgressSnapshot(
        step: 0,
        stepCount: 0,
        fraction: 0,
        iterationsPerSecond: nil,
        etaSeconds: nil,
        elapsedSeconds: 0,
        phaseText: "Idle"
    )
}

/// Scheduler selector for the app UI. Only discreteFlowScheduler is used by Z-Image.
public enum StableDiffusionScheduler: String {
    case pndmScheduler
    case dpmSolverMultistepScheduler
    case discreteFlowScheduler
}

class GenerationContext: ObservableObject {
    let scheduler = StableDiffusionScheduler.discreteFlowScheduler

    @Published var pipeline: AppPipeline? = nil {
        didSet {
            if let pipeline = pipeline {
                progressSubscriber = pipeline
                    .progressPublisher
                    .receive(on: DispatchQueue.main)
                    .sink { progress in
                        guard let progress = progress else { return }
                        self.updatePreviewIfNeeded(progress)
                        self.updateGenerationProgressTracking(progress)
                        self.state = .running(progress)
                    }
            }
        }
    }
    @Published var state: GenerationState = .startup {
        didSet {
            if case .running = state {
                return
            }
            resetGenerationProgressTracking()
        }
    }
    
    @Published var positivePrompt = Settings.shared.prompt
    @Published var negativePrompt = Settings.shared.negativePrompt

    // FIXME: Double to support the slider component
    @Published var steps: Double = Settings.shared.stepCount
    @Published var numImages: Double = 1.0
    @Published var seed: UInt32 = Settings.shared.seed
    @Published var guidanceScale: Double = Settings.shared.guidanceScale
    @Published var previews: Double = runningOnMac ? Settings.shared.previewCount : 0.0
    @Published var disableSafety = false
    @Published var previewImage: CGImage? = nil
    @Published var variationAmount: Double = 1.0
    @Published var variationBaseSeed: UInt32? = nil
    @Published var variationBaseImage: CGImage? = nil
    @Published var variationBaseNoiseData: Data? = nil
    @Published var variationBaseNoiseShape: [Int]? = nil

    @Published var computeUnits: ComputeUnits = Settings.shared.userSelectedComputeUnits ?? .cpuOnly
    @Published var transformerModelPath: String? = Settings.shared.transformerModelPath
    @Published var vaeDecoderPath: String? = Settings.shared.vaeDecoderPath
    @Published var externalEmbeddingsPath: String? = Settings.shared.externalEmbeddingsPath
    @Published var externalLoRAPath: String? = Settings.shared.externalLoRAPath
    @Published var initialLatentPath: String? = Settings.shared.initialLatentPath
    @Published var generationProgressSnapshot: GenerationProgressSnapshot = .idle

    private var progressSubscriber: Cancellable?
    private var generationProgressStartDate: Date?
    private var generationProgressLastDate: Date?
    private var generationProgressLastStep: Int?
    private var generationProgressSmoothedStepSeconds: Double?
    private let etaSmoothingAlpha = 0.25
    private let initialLatentShapeContract = [1, 16, 64, 64]

    init() {
        migrateLegacyDesktopDefaultPaths()
        #if os(iOS)
        clearStaleIOSResourcePaths()
        #endif
    }

    private func migrateLegacyDesktopDefaultPaths() {
        let legacyDocumentsRoot = "/Users/a1111/Documents"

        func isLegacyTransformerPath(_ path: String?) -> Bool {
            guard let path else { return false }
            return path.hasPrefix("\(legacyDocumentsRoot)/ZImageTurbo_TransformerBackbone_stage")
        }

        func isLegacyLoRAPath(_ path: String?) -> Bool {
            path == "\(legacyDocumentsRoot)/z_image_lora.safetensors"
        }

        func resolvedBookmarkPath(_ bookmarkData: Data?) -> String? {
            guard let bookmarkData else { return nil }
            var isStale = false
            do {
                #if os(macOS)
                let resolvedURL = try URL(
                    resolvingBookmarkData: bookmarkData,
                    options: [.withSecurityScope],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
                #else
                let resolvedURL = try URL(
                    resolvingBookmarkData: bookmarkData,
                    options: [],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
                #endif
                return resolvedURL.path
            } catch {
                return nil
            }
        }

        let transformerBookmarkPath = resolvedBookmarkPath(Settings.shared.transformerModelBookmark)
        if isLegacyTransformerPath(transformerModelPath) || isLegacyTransformerPath(transformerBookmarkPath) {
            transformerModelPath = nil
            Settings.shared.transformerModelPath = nil
            Settings.shared.transformerModelBookmark = nil
        }

        let loraBookmarkPath = resolvedBookmarkPath(Settings.shared.externalLoRABookmark)
        if isLegacyLoRAPath(externalLoRAPath) || isLegacyLoRAPath(loraBookmarkPath) {
            externalLoRAPath = nil
            Settings.shared.externalLoRAPath = nil
            Settings.shared.externalLoRABookmark = nil
        }
    }

    /// On iOS, any path saved from a previous (failed) file-picker session may be
    /// sitting in UserDefaults pointing at a file that no longer exists.  That stale
    /// path takes priority over the automatic Documents-folder scan and makes the
    /// app appear to be broken even when the user has already placed their files in
    /// the right location.  This method removes every path whose target is absent so
    /// the fallback logic in defaultResourceURL() can do its job.
    #if os(iOS)
    func clearStaleIOSResourcePaths() {
        let fm = FileManager.default

        func clearIfMissing(path: String?, clearAction: () -> Void) {
            guard let p = path, !p.isEmpty else { return }
            if !fm.fileExists(atPath: p) { clearAction() }
        }

        clearIfMissing(path: transformerModelPath) {
            transformerModelPath = nil
            Settings.shared.transformerModelPath = nil
            Settings.shared.transformerModelBookmark = nil
        }
        clearIfMissing(path: vaeDecoderPath) {
            vaeDecoderPath = nil
            Settings.shared.vaeDecoderPath = nil
            Settings.shared.vaeDecoderBookmark = nil
        }
        clearIfMissing(path: externalEmbeddingsPath) {
            externalEmbeddingsPath = nil
            Settings.shared.externalEmbeddingsPath = nil
            Settings.shared.externalEmbeddingsBookmark = nil
        }
        clearIfMissing(path: externalLoRAPath) {
            externalLoRAPath = nil
            Settings.shared.externalLoRAPath = nil
            Settings.shared.externalLoRABookmark = nil
        }
        clearIfMissing(path: initialLatentPath) {
            initialLatentPath = nil
            Settings.shared.initialLatentPath = nil
            Settings.shared.initialLatentBookmark = nil
        }
    }
    #endif

    private func updatePreviewIfNeeded(_ progress: StableDiffusionProgress) {
        if previews == 0 || progress.step == 0 {
            previewImage = nil
        }

        if previews > 0, let newImage = progress.currentImages.first, newImage != nil {
            previewImage = newImage
        }
    }

    private func applyVariationBase(seed: UInt32, image: CGImage?, noiseData: Data?, noiseShape: [Int]?) {
        variationBaseSeed = seed
        variationBaseImage = image
        variationBaseNoiseData = noiseData
        variationBaseNoiseShape = noiseShape
    }

    func updateVariationBase(seed: UInt32, image: CGImage?, noiseData: Data? = nil, noiseShape: [Int]? = nil) {
        if Thread.isMainThread {
            applyVariationBase(seed: seed, image: image, noiseData: noiseData, noiseShape: noiseShape)
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.applyVariationBase(seed: seed, image: image, noiseData: noiseData, noiseShape: noiseShape)
            }
        }
    }

    func loadHistorySelection(
        prompt: String,
        seed: UInt32,
        image: CGImage?,
        noiseData: Data?,
        noiseShape: [Int]?
    ) {
        positivePrompt = prompt
        Settings.shared.prompt = prompt
        updateVariationBase(seed: seed, image: image, noiseData: noiseData, noiseShape: noiseShape)
        resetGenerationProgressTracking()
        state = .complete(prompt, image, seed, nil)
    }

    @MainActor
    func generate(
        prompt overridePrompt: String? = nil,
        baseSeed: UInt32? = nil,
        baseImage _: CGImage? = nil,
        forceSeed: UInt32? = nil
    ) async throws -> GenerationResult {
        guard let pipeline = pipeline else { throw "No pipeline" }
        // All setup runs on the main actor — safe to read/write @Published properties.
        beginGenerationProgressTracking(estimatedStepCount: Int(steps))
        let variation = max(0, min(1, variationAmount))
        let sourceSeed = baseSeed ?? variationBaseSeed
        let sourceNoiseData = variationBaseNoiseData
        let sourceNoiseShape = variationBaseNoiseShape
        let configuredSeed = forceSeed ?? seed

        var generationSeed = configuredSeed
        var initialNoiseData: Data? = nil
        var initialNoiseShape: [Int]? = nil
        var interpolationBaseNoiseData: Data? = nil
        var interpolationBaseNoiseShape: [Int]? = nil
        var interpolationSeed: UInt32? = nil
        var interpolationAmount: Float? = nil

        if let configuredInitialLatentData = try loadConfiguredInitialLatentData() {
            initialNoiseData = configuredInitialLatentData
            initialNoiseShape = initialLatentShapeContract
        } else {
            if variation <= 0 {
                if let sourceNoiseData, let sourceNoiseShape {
                    initialNoiseData = sourceNoiseData
                    initialNoiseShape = sourceNoiseShape
                } else if let sourceSeed {
                    generationSeed = sourceSeed
                }
            } else if variation < 1 {
                if configuredSeed == 0 {
                    generationSeed = UInt32.random(in: 1...UInt32.max)
                }
                if let sourceNoiseData, let sourceNoiseShape {
                    interpolationBaseNoiseData = sourceNoiseData
                    interpolationBaseNoiseShape = sourceNoiseShape
                    interpolationAmount = Float(variation)
                } else if let sourceSeed {
                    interpolationSeed = sourceSeed
                    interpolationAmount = Float(variation)
                }
            } else if configuredSeed == 0 {
                generationSeed = UInt32.random(in: 1...UInt32.max)
            }
        }

        // Capture all @Published values now (on main actor) so the background
        // thread never touches self's properties.
        let capturedPrompt = overridePrompt ?? positivePrompt
        let capturedNegativePrompt = negativePrompt
        let capturedScheduler = scheduler
        let capturedSteps = Int(steps)
        let capturedPreviews = Int(previews)
        let capturedGuidanceScale = Float(guidanceScale)
        let capturedDisableSafety = disableSafety

        // Run the blocking CoreML inference on a background thread so the main
        // thread (and layout engine) are never blocked.
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try pipeline.generate(
                        prompt: capturedPrompt,
                        negativePrompt: capturedNegativePrompt,
                        scheduler: capturedScheduler,
                        numInferenceSteps: capturedSteps,
                        seed: generationSeed,
                        numPreviews: capturedPreviews,
                        guidanceScale: capturedGuidanceScale,
                        disableSafety: capturedDisableSafety,
                        startingImage: nil,
                        strength: nil,
                        initialNoiseData: initialNoiseData,
                        initialNoiseShape: initialNoiseShape,
                        interpolationBaseNoiseData: interpolationBaseNoiseData,
                        interpolationBaseNoiseShape: interpolationBaseNoiseShape,
                        interpolationSeed: interpolationSeed,
                        interpolationAmount: interpolationAmount
                    )
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    func cancelGeneration() {
        pipeline?.setCancelled()
        resetGenerationProgressTracking()
    }

    private func beginGenerationProgressTracking(estimatedStepCount: Int) {
        let now = Date()
        generationProgressStartDate = now
        generationProgressLastDate = now
        generationProgressLastStep = nil
        generationProgressSmoothedStepSeconds = nil
        generationProgressSnapshot = GenerationProgressSnapshot(
            step: 0,
            stepCount: max(0, estimatedStepCount),
            fraction: 0,
            iterationsPerSecond: nil,
            etaSeconds: nil,
            elapsedSeconds: 0,
            phaseText: "Preparing model… ETA pending"
        )
    }

    private func resetGenerationProgressTracking() {
        generationProgressStartDate = nil
        generationProgressLastDate = nil
        generationProgressLastStep = nil
        generationProgressSmoothedStepSeconds = nil
        generationProgressSnapshot = .idle
    }

    private func updateGenerationProgressTracking(_ progress: StableDiffusionProgress) {
        let now = Date()
        if generationProgressStartDate == nil {
            beginGenerationProgressTracking(estimatedStepCount: Int(progress.stepCount))
        }
        guard let startDate = generationProgressStartDate else {
            return
        }

        let elapsed = now.timeIntervalSince(startDate)
        let stepCount = max(Int(progress.stepCount), 0)
        let step = stepCount > 0 ? max(1, min(stepCount, Int(progress.step) + 1)) : 0

        if let previousStep = generationProgressLastStep,
           let previousDate = generationProgressLastDate,
           step > previousStep {
            let secondsPerStep = max(now.timeIntervalSince(previousDate), 0.0001)
            if let existing = generationProgressSmoothedStepSeconds {
                generationProgressSmoothedStepSeconds =
                    (etaSmoothingAlpha * secondsPerStep) + ((1 - etaSmoothingAlpha) * existing)
            } else {
                generationProgressSmoothedStepSeconds = secondsPerStep
            }
            generationProgressLastDate = now
            generationProgressLastStep = step
        } else if generationProgressLastStep == nil {
            generationProgressLastDate = now
            generationProgressLastStep = step
        }

        let fraction = stepCount > 0 ? Double(step) / Double(stepCount) : 0
        let smoothed = generationProgressSmoothedStepSeconds
        let iterationsPerSecond = smoothed.map { 1.0 / $0 }
        let remaining = max(stepCount - step, 0)
        let etaSeconds = smoothed.map { $0 * Double(remaining) }
        let phaseText: String
        if stepCount == 0 {
            phaseText = "Preparing model… ETA pending"
        } else if step <= 1 && iterationsPerSecond == nil {
            phaseText = "Preparing first denoising step… ETA pending"
        } else {
            phaseText = "Denoising"
        }

        generationProgressSnapshot = GenerationProgressSnapshot(
            step: step,
            stepCount: stepCount,
            fraction: fraction,
            iterationsPerSecond: iterationsPerSecond,
            etaSeconds: etaSeconds,
            elapsedSeconds: elapsed,
            phaseText: phaseText
        )
    }

    var externalEmbeddingsURL: URL? {
        guard let path = externalEmbeddingsPath?.trimmingCharacters(in: .whitespacesAndNewlines),
              !path.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: path)
    }

    var transformerModelURL: URL {
        resolvePathWithDetail(
            path: transformerModelPath,
            bookmarkData: Settings.shared.transformerModelBookmark,
            refreshBookmark: { Settings.shared.transformerModelBookmark = $0 },
            bundledFallbackName: "ZImageTurbo_TransformerBackbone_stage0.mlmodelc",
            resourceLabel: "Transformer"
        ).url
    }

    var transformerStageURLs: [URL] {
        let resolved = transformerModelURL
        let baseDir = resolved.deletingLastPathComponent()
        let fileName = resolved.deletingPathExtension().lastPathComponent
        let ext = resolved.pathExtension.isEmpty ? "mlmodelc" : resolved.pathExtension

        if let range = fileName.range(of: "_stage\\d+$", options: .regularExpression) {
            let prefix = String(fileName[..<range.lowerBound]) + "_stage"
            let fm = FileManager.default
            var stages: [(index: Int, url: URL)] = []
            for i in 0..<100 {
                let candidate = baseDir.appending(path: "\(prefix)\(i).\(ext)")
                guard fm.fileExists(atPath: candidate.path) else { break }
                stages.append((i, candidate))
            }
            if !stages.isEmpty {
                return stages.sorted(by: { $0.index < $1.index }).map(\.url)
            }
        }

        return ZImageCheckpointSet.defaultTransformerStageNames.map {
            baseDir.appending(path: $0)
        }
    }

    var vaeDecoderModelURL: URL {
        resolvePathWithDetail(
            path: vaeDecoderPath,
            bookmarkData: Settings.shared.vaeDecoderBookmark,
            refreshBookmark: { Settings.shared.vaeDecoderBookmark = $0 },
            bundledFallbackName: "VAEDecoder.mlmodelc",
            resourceLabel: "VAE"
        ).url
    }

    var externalEmbeddingsFileURL: URL? {
        resolveOptionalPathWithDetail(
            path: externalEmbeddingsPath,
            bookmarkData: Settings.shared.externalEmbeddingsBookmark,
            refreshBookmark: { Settings.shared.externalEmbeddingsBookmark = $0 },
            resourceLabel: "Embeddings"
        ).url
    }

    var effectiveEmbeddingsURL: URL {
        externalEmbeddingsFileURL ?? defaultResourceURL(named: "zimage_embeddings.bin")
    }

    var externalLoRAFileURL: URL? {
        resolveOptionalPathWithDetail(
            path: externalLoRAPath,
            bookmarkData: Settings.shared.externalLoRABookmark,
            refreshBookmark: { Settings.shared.externalLoRABookmark = $0 },
            resourceLabel: "LoRA"
        ).url
    }

    var effectiveLoRAURL: URL? {
        externalLoRAFileURL ?? defaultResourceURL(named: "z_image_lora.safetensors")
    }


    var transformerPathResolutionDetail: String {
        resolvePathWithDetail(
            path: transformerModelPath,
            bookmarkData: Settings.shared.transformerModelBookmark,
            refreshBookmark: { Settings.shared.transformerModelBookmark = $0 },
            bundledFallbackName: "ZImageTurbo_TransformerBackbone_stage0.mlmodelc",
            resourceLabel: "Transformer"
        ).detail
    }

    var vaePathResolutionDetail: String {
        resolvePathWithDetail(
            path: vaeDecoderPath,
            bookmarkData: Settings.shared.vaeDecoderBookmark,
            refreshBookmark: { Settings.shared.vaeDecoderBookmark = $0 },
            bundledFallbackName: "VAEDecoder.mlmodelc",
            resourceLabel: "VAE"
        ).detail
    }

    var embeddingsPathResolutionDetail: String {
        if externalEmbeddingsPath?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true {
            let fallback = defaultResourceURL(named: "zimage_embeddings.bin")
            return "Embeddings: no external path configured; using default resource (\(fallback.path))."
        }
        return resolveOptionalPathWithDetail(
            path: externalEmbeddingsPath,
            bookmarkData: Settings.shared.externalEmbeddingsBookmark,
            refreshBookmark: { Settings.shared.externalEmbeddingsBookmark = $0 },
            resourceLabel: "Embeddings"
        ).detail
    }

    var loraPathResolutionDetail: String {
        if externalLoRAPath?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true {
            let fallback = defaultResourceURL(named: "z_image_lora.safetensors")
            return "LoRA: no external path configured; using default resource (\(fallback.path))."
        }
        return resolveOptionalPathWithDetail(
            path: externalLoRAPath,
            bookmarkData: Settings.shared.externalLoRABookmark,
            refreshBookmark: { Settings.shared.externalLoRABookmark = $0 },
            resourceLabel: "LoRA"
        ).detail
    }

    var initialLatentFileURL: URL? {
        resolveOptionalPathWithDetail(
            path: initialLatentPath,
            bookmarkData: Settings.shared.initialLatentBookmark,
            refreshBookmark: { Settings.shared.initialLatentBookmark = $0 },
            resourceLabel: "Initial latent"
        ).url
    }

    var initialLatentPathResolutionDetail: String {
        resolveOptionalPathWithDetail(
            path: initialLatentPath,
            bookmarkData: Settings.shared.initialLatentBookmark,
            refreshBookmark: { Settings.shared.initialLatentBookmark = $0 },
            resourceLabel: "Initial latent"
        ).detail
    }

    func logResolvedModelPaths() {
        for (index, url) in transformerStageURLs.enumerated() {
            print("[ZImagePaths] transformerStage[\(index)]=\(url.path)")
        }
        print("[ZImagePaths] vaeDecoder=\(vaeDecoderModelURL.path)")
        print("[ZImagePaths] textEmbeddings=\(effectiveEmbeddingsURL.path)")
        print("[ZImagePaths] lora=\(effectiveLoRAURL?.path ?? "<none>")")
    }

    func setTransformerModelURL(_ url: URL?) {
        let normalized = url?.path.trimmingCharacters(in: .whitespacesAndNewlines)
        transformerModelPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.transformerModelPath = transformerModelPath
        Settings.shared.transformerModelBookmark = makeBookmark(for: url)
    }

    /// Sets the transformer path from an in-container copy; clears any stale bookmark.
    func setTransformerModelPath(_ path: String?) {
        let normalized = path?.trimmingCharacters(in: .whitespacesAndNewlines)
        transformerModelPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.transformerModelPath = transformerModelPath
        Settings.shared.transformerModelBookmark = nil
    }

    func setVaeDecoderModelURL(_ url: URL?) {
        let normalized = url?.path.trimmingCharacters(in: .whitespacesAndNewlines)
        vaeDecoderPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.vaeDecoderPath = vaeDecoderPath
        Settings.shared.vaeDecoderBookmark = makeBookmark(for: url)
    }

    /// Sets the VAE path from an in-container copy; clears any stale bookmark.
    func setVaeDecoderModelPath(_ path: String?) {
        let normalized = path?.trimmingCharacters(in: .whitespacesAndNewlines)
        vaeDecoderPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.vaeDecoderPath = vaeDecoderPath
        Settings.shared.vaeDecoderBookmark = nil
    }

    func setExternalEmbeddingsURL(_ url: URL?) {
        let normalized = url?.path.trimmingCharacters(in: .whitespacesAndNewlines)
        externalEmbeddingsPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.externalEmbeddingsPath = externalEmbeddingsPath
        Settings.shared.externalEmbeddingsBookmark = makeBookmark(for: url)
    }

    func setExternalEmbeddingsPath(_ path: String?) {
        let normalized = path?.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalValue = (normalized?.isEmpty ?? true) ? nil : normalized
        externalEmbeddingsPath = finalValue
        Settings.shared.externalEmbeddingsPath = finalValue
        Settings.shared.externalEmbeddingsBookmark = nil
    }

    func setExternalLoRAURL(_ url: URL?) {
        let normalized = url?.path.trimmingCharacters(in: .whitespacesAndNewlines)
        externalLoRAPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.externalLoRAPath = externalLoRAPath
        Settings.shared.externalLoRABookmark = makeBookmark(for: url)
    }

    func setExternalLoRAPath(_ path: String?) {
        let normalized = path?.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalValue = (normalized?.isEmpty ?? true) ? nil : normalized
        externalLoRAPath = finalValue
        Settings.shared.externalLoRAPath = finalValue
        Settings.shared.externalLoRABookmark = nil
    }

    func setInitialLatentURL(_ url: URL?) {
        let normalized = url?.path.trimmingCharacters(in: .whitespacesAndNewlines)
        initialLatentPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.initialLatentPath = initialLatentPath
        Settings.shared.initialLatentBookmark = makeBookmark(for: url)
    }

    /// Sets the initial latent path from an in-container copy; clears any stale bookmark.
    func setInitialLatentPath(_ path: String?) {
        let normalized = path?.trimmingCharacters(in: .whitespacesAndNewlines)
        initialLatentPath = (normalized?.isEmpty ?? true) ? nil : normalized
        Settings.shared.initialLatentPath = initialLatentPath
        Settings.shared.initialLatentBookmark = nil
    }

    private func makeBookmark(for url: URL?) -> Data? {
        guard let url else { return nil }
        do {
            #if os(macOS)
            return try url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
            #else
            return try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
            #endif
        } catch {
            return nil
        }
    }

    private func resolvePathWithDetail(
        path: String?,
        bookmarkData: Data?,
        refreshBookmark: (Data?) -> Void,
        bundledFallbackName: String,
        resourceLabel: String
    ) -> (url: URL, detail: String) {
        let fallbackURL = defaultResourceURL(named: bundledFallbackName)
        if let bookmarkData {
            var isStale = false
            do {
                #if os(macOS)
                let resolvedURL = try URL(
                    resolvingBookmarkData: bookmarkData,
                    options: [.withSecurityScope],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
                #else
                let resolvedURL = try URL(
                    resolvingBookmarkData: bookmarkData,
                    options: [],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
                #endif
                if isStale {
                    refreshBookmark(makeBookmark(for: resolvedURL))
                    return (resolvedURL, "\(resourceLabel): using refreshed security-scoped bookmark (\(resolvedURL.path)).")
                }
                return (resolvedURL, "\(resourceLabel): using security-scoped bookmark (\(resolvedURL.path)).")
            } catch {
                if let path = path?.trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty {
                    return (
                        URL(fileURLWithPath: path),
                        "\(resourceLabel): bookmark resolution failed (\(error.localizedDescription)); falling back to stored path (\(path))."
                    )
                }
                return (
                    fallbackURL,
                    "\(resourceLabel): bookmark resolution failed (\(error.localizedDescription)); falling back to default resource (\(fallbackURL.path))."
                )
            }
        }
        guard let path = path?.trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty else {
            return (
                fallbackURL,
                "\(resourceLabel): no external path configured; using default resource (\(fallbackURL.path))."
            )
        }
        return (URL(fileURLWithPath: path), "\(resourceLabel): using stored external path (\(path)).")
    }

    private func resolveOptionalPathWithDetail(
        path: String?,
        bookmarkData: Data?,
        refreshBookmark: (Data?) -> Void,
        resourceLabel: String
    ) -> (url: URL?, detail: String) {
        if let bookmarkData {
            var isStale = false
            do {
                #if os(macOS)
                let resolvedURL = try URL(
                    resolvingBookmarkData: bookmarkData,
                    options: [.withSecurityScope],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
                #else
                let resolvedURL = try URL(
                    resolvingBookmarkData: bookmarkData,
                    options: [],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
                #endif
                if isStale {
                    refreshBookmark(makeBookmark(for: resolvedURL))
                    return (resolvedURL, "\(resourceLabel): using refreshed security-scoped bookmark (\(resolvedURL.path)).")
                }
                return (resolvedURL, "\(resourceLabel): using security-scoped bookmark (\(resolvedURL.path)).")
            } catch {
                if let path = path?.trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty {
                    return (
                        URL(fileURLWithPath: path),
                        "\(resourceLabel): bookmark resolution failed (\(error.localizedDescription)); falling back to stored path (\(path))."
                    )
                }
                return (nil, "\(resourceLabel): bookmark resolution failed (\(error.localizedDescription)); path is not configured.")
            }
        }
        guard let path = path?.trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty else {
            return (nil, "\(resourceLabel): no external path configured.")
        }
        return (URL(fileURLWithPath: path), "\(resourceLabel): using stored external path (\(path)).")
    }

    private func loadConfiguredInitialLatentData() throws -> Data? {
        guard let latentURL = initialLatentFileURL else {
            return nil
        }

        let accessGranted = latentURL.startAccessingSecurityScopedResource()
        defer {
            if accessGranted {
                latentURL.stopAccessingSecurityScopedResource()
            }
        }

        let latentData = try Data(contentsOf: latentURL)
        let expectedFloatCount = initialLatentShapeContract.reduce(1, *)
        let expectedByteCount = expectedFloatCount * MemoryLayout<Float32>.size
        guard latentData.count == expectedByteCount else {
            throw InitialLatentFileError(
                message: "Initial latent byte count mismatch. Expected \(expectedByteCount), got \(latentData.count). Path: \(latentURL.path)"
            )
        }
        return latentData
    }

    private func bundledResourceURL(named resourceName: String) -> URL {
        guard let resourceURL = Bundle.main.resourceURL else {
            return URL(fileURLWithPath: resourceName)
        }
        return resourceURL.appending(path: resourceName)
    }

    private var hardcodedDesktopTransformerBaseURL: URL {
        URL(
            fileURLWithPath: "/Users/a1111/Downloads/zimage_true_stage_fp32_7stage_lora_as_input_activation_space/",
            isDirectory: true
        )
    }

    private var hardcodedDesktopLoRAURL: URL {
        URL(fileURLWithPath: "/Users/a1111/Downloads/z_image_lora.safetensors")
    }

    private func hardcodedDesktopDefaultResourceURL(named resourceName: String) -> URL? {
        if resourceName.hasPrefix("ZImageTurbo_TransformerBackbone_stage") {
            return hardcodedDesktopTransformerBaseURL.appendingPathComponent(resourceName, isDirectory: true)
        }
        if resourceName == "z_image_lora.safetensors" {
            return hardcodedDesktopLoRAURL
        }
        return nil
    }

    private var hardcodedIOSDefaultCheckpointBaseURL: URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents", isDirectory: true)
        return documentsURL
            .appendingPathComponent("zimage-checkpoints", isDirectory: true)
            .appendingPathComponent("default", isDirectory: true)
    }

    private func defaultResourceURL(named resourceName: String) -> URL {
        #if os(iOS)
        if runningOnMac, let hardcodedURL = hardcodedDesktopDefaultResourceURL(named: resourceName) {
            return hardcodedURL
        }
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents", isDirectory: true)

        // Check Documents root first — user simply drops files into
        // On My iPhone > [App Name] via the Files app.
        let rootURL = docs.appendingPathComponent(resourceName)
        if fm.fileExists(atPath: rootURL.path) { return rootURL }

        // Also check the Documents/zimage-checkpoints/default/ subfolder.
        let subURL = hardcodedIOSDefaultCheckpointBaseURL.appendingPathComponent(resourceName)
        if fm.fileExists(atPath: subURL.path) { return subURL }

        // Neither found — return the Documents-root path so any error message
        // shows the user a clear, recognisable location.
        return rootURL
        #else
        if let hardcodedURL = hardcodedDesktopDefaultResourceURL(named: resourceName) {
            return hardcodedURL
        }
        return bundledResourceURL(named: resourceName)
        #endif
    }

}

class Settings {
    static let shared = Settings()
    
    let defaults = UserDefaults.standard
    
    enum Keys: String {
        case model
        case safetyCheckerDisclaimer
        case computeUnits
        case prompt
        case negativePrompt
        case guidanceScale
        case stepCount
        case previewCount
        case seed
        case transformerModelPath
        case transformerModelBookmark
        case vaeDecoderPath
        case vaeDecoderBookmark
        case externalEmbeddingsPath
        case externalEmbeddingsBookmark
        case externalLoRAPath
        case externalLoRABookmark
        case initialLatentPath
        case initialLatentBookmark
    }

    private init() {
        defaults.register(defaults: [
            Keys.model.rawValue: ModelInfo.v2Base.modelId,
            Keys.safetyCheckerDisclaimer.rawValue: false,
            Keys.computeUnits.rawValue: -1,      // Use default
            Keys.prompt.rawValue: DEFAULT_PROMPT,
            Keys.negativePrompt.rawValue: "",
            Keys.guidanceScale.rawValue: 0.0,
            Keys.stepCount.rawValue: 4,
            Keys.previewCount.rawValue: 0,
            Keys.seed.rawValue: 0
        ])
    }

    var currentModel: ModelInfo {
        set {
            defaults.set(newValue.modelId, forKey: Keys.model.rawValue)
        }
        get {
            guard let modelId = defaults.string(forKey: Keys.model.rawValue) else { return DEFAULT_MODEL }
            return ModelInfo.from(modelId: modelId) ?? DEFAULT_MODEL
        }
    }

    var prompt: String {
        set {
            defaults.set(newValue, forKey: Keys.prompt.rawValue)
        }
        get {
            return defaults.string(forKey: Keys.prompt.rawValue) ?? DEFAULT_PROMPT
        }
    }

    var negativePrompt: String {
        set {
            defaults.set(newValue, forKey: Keys.negativePrompt.rawValue)
        }
        get {
            return defaults.string(forKey: Keys.negativePrompt.rawValue) ?? ""
        }
    }

    var guidanceScale: Double {
        set {
            defaults.set(newValue, forKey: Keys.guidanceScale.rawValue)
        }
        get {
            return defaults.double(forKey: Keys.guidanceScale.rawValue)
        }
    }

    var stepCount: Double {
        set {
            defaults.set(newValue, forKey: Keys.stepCount.rawValue)
        }
        get {
            return defaults.double(forKey: Keys.stepCount.rawValue)
        }
    }

    var previewCount: Double {
        set {
            defaults.set(newValue, forKey: Keys.previewCount.rawValue)
        }
        get {
            return defaults.double(forKey: Keys.previewCount.rawValue)
        }
    }

    var seed: UInt32 {
        set {
            defaults.set(String(newValue), forKey: Keys.seed.rawValue)
        }
        get {
            if let seedString = defaults.string(forKey: Keys.seed.rawValue), let seedValue = UInt32(seedString) {
                return seedValue
            }
            return 0
        }
    }

    var externalEmbeddingsPath: String? {
        set {
            if let value = newValue, !value.isEmpty {
                defaults.set(value, forKey: Keys.externalEmbeddingsPath.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.externalEmbeddingsPath.rawValue)
            }
        }
        get {
            defaults.string(forKey: Keys.externalEmbeddingsPath.rawValue)
        }
    }

    var externalLoRAPath: String? {
        set {
            if let value = newValue, !value.isEmpty {
                defaults.set(value, forKey: Keys.externalLoRAPath.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.externalLoRAPath.rawValue)
            }
        }
        get {
            defaults.string(forKey: Keys.externalLoRAPath.rawValue)
        }
    }

    var transformerModelPath: String? {
        set {
            if let value = newValue, !value.isEmpty {
                defaults.set(value, forKey: Keys.transformerModelPath.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.transformerModelPath.rawValue)
            }
        }
        get {
            defaults.string(forKey: Keys.transformerModelPath.rawValue)
        }
    }

    var transformerModelBookmark: Data? {
        set {
            if let newValue {
                defaults.set(newValue, forKey: Keys.transformerModelBookmark.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.transformerModelBookmark.rawValue)
            }
        }
        get {
            defaults.data(forKey: Keys.transformerModelBookmark.rawValue)
        }
    }

    var vaeDecoderPath: String? {
        set {
            if let value = newValue, !value.isEmpty {
                defaults.set(value, forKey: Keys.vaeDecoderPath.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.vaeDecoderPath.rawValue)
            }
        }
        get {
            defaults.string(forKey: Keys.vaeDecoderPath.rawValue)
        }
    }

    var vaeDecoderBookmark: Data? {
        set {
            if let newValue {
                defaults.set(newValue, forKey: Keys.vaeDecoderBookmark.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.vaeDecoderBookmark.rawValue)
            }
        }
        get {
            defaults.data(forKey: Keys.vaeDecoderBookmark.rawValue)
        }
    }

    var externalEmbeddingsBookmark: Data? {
        set {
            if let newValue {
                defaults.set(newValue, forKey: Keys.externalEmbeddingsBookmark.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.externalEmbeddingsBookmark.rawValue)
            }
        }
        get {
            defaults.data(forKey: Keys.externalEmbeddingsBookmark.rawValue)
        }
    }

    var externalLoRABookmark: Data? {
        set {
            if let newValue {
                defaults.set(newValue, forKey: Keys.externalLoRABookmark.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.externalLoRABookmark.rawValue)
            }
        }
        get {
            defaults.data(forKey: Keys.externalLoRABookmark.rawValue)
        }
    }

    var initialLatentPath: String? {
        set {
            if let value = newValue, !value.isEmpty {
                defaults.set(value, forKey: Keys.initialLatentPath.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.initialLatentPath.rawValue)
            }
        }
        get {
            defaults.string(forKey: Keys.initialLatentPath.rawValue)
        }
    }

    var initialLatentBookmark: Data? {
        set {
            if let newValue {
                defaults.set(newValue, forKey: Keys.initialLatentBookmark.rawValue)
            } else {
                defaults.removeObject(forKey: Keys.initialLatentBookmark.rawValue)
            }
        }
        get {
            defaults.data(forKey: Keys.initialLatentBookmark.rawValue)
        }
    }

    var safetyCheckerDisclaimerShown: Bool {
        set {
            defaults.set(newValue, forKey: Keys.safetyCheckerDisclaimer.rawValue)
        }
        get {
            return defaults.bool(forKey: Keys.safetyCheckerDisclaimer.rawValue)
        }
    }
    
    /// Returns the option selected by the user, if overridden
    /// `nil` means: guess best
    var userSelectedComputeUnits: ComputeUnits? {
        set {
            // Any value other than the supported ones would cause `get` to return `nil`
            defaults.set(newValue?.rawValue ?? -1, forKey: Keys.computeUnits.rawValue)
        }
        get {
            let current = defaults.integer(forKey: Keys.computeUnits.rawValue)
            guard current != -1 else { return nil }
            return ComputeUnits(rawValue: current)
        }
    }

    /// Folder inside Application Support where user-imported resources are copied.
    /// Files here are in the app container and need no security scope.
    public func importedResourcesURL() -> URL {
        let url = applicationSupportURL()
            .appendingPathComponent("zimage-imported-resources", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    public func applicationSupportURL() -> URL {
        let fileManager = FileManager.default
        guard let appDirectoryURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            // To ensure we don't return an optional - if the user domain application support cannot be accessed use the top level application support directory
            return URL.applicationSupportDirectory
        }

        do {
            // Create the application support directory if it doesn't exist
            try fileManager.createDirectory(at: appDirectoryURL, withIntermediateDirectories: true, attributes: nil)
            return appDirectoryURL
        } catch {
            print("Error creating application support directory: \(error)")
            return fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        }
    }

    func tempStorageURL() -> URL {
        
        let tmpDir = applicationSupportURL().appendingPathComponent("hf-diffusion-tmp")
        
        // Create directory if it doesn't exist
        if !FileManager.default.fileExists(atPath: tmpDir.path) {
            do {
                try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Failed to create temporary directory: \(error)")
                return FileManager.default.temporaryDirectory
            }
        }
        
        return tmpDir
    }

}
