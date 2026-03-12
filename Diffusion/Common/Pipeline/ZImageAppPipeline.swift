import Combine
import CoreGraphics
import Foundation
import StableDiffusion
#if canImport(UIKit)
import UIKit
#endif

@available(iOS 17.0, macOS 14.0, *)
final class ZImageAppPipeline: AppPipeline {
    private var pipeline: StableDiffusion.ZImagePipeline
    private let transformerStageURLs: [URL]
    private let vaeDecoderURL: URL
    private let embeddingsURL: URL
    private var canceled = false
    private let fileManager = FileManager.default
    private var memoryWarningObserver: NSObjectProtocol?

    var progressPublisher: CurrentValueSubject<StableDiffusionProgress?, Never> = .init(nil)

    init(
        pipeline: StableDiffusion.ZImagePipeline,
        transformerStageURLs: [URL],
        vaeDecoderURL: URL,
        embeddingsURL: URL
    ) {
        self.pipeline = pipeline
        self.transformerStageURLs = transformerStageURLs
        self.vaeDecoderURL = vaeDecoderURL
        self.embeddingsURL = embeddingsURL
        
        setupMemoryWarningObserver()
    }
    
    deinit {
        if let observer = memoryWarningObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }
    
    private func setupMemoryWarningObserver() {
        #if canImport(UIKit) && !os(watchOS)
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
        #endif
    }
    
    private func handleMemoryWarning() {
        print("[ZImageAppPipeline] Received memory warning - unloading resources")
        pipeline.unloadResources()
        print("[ZImageAppPipeline] Memory warning handled - resources unloaded")
    }

    func generate(
        prompt _: String,
        negativePrompt _: String,
        scheduler _: StableDiffusionScheduler,
        numInferenceSteps stepCount: Int = 4,
        seed: UInt32 = 0,
        numPreviews _: Int = 0,
        guidanceScale _: Float = 0,
        disableSafety _: Bool = true,
        startingImage _: CGImage? = nil,
        strength _: Float? = nil,
        initialNoiseData: Data? = nil,
        initialNoiseShape: [Int]? = nil,
        interpolationBaseNoiseData _: Data? = nil,
        interpolationBaseNoiseShape _: [Int]? = nil,
        interpolationSeed _: UInt32? = nil,
        interpolationAmount _: Float? = nil
    ) throws -> GenerationResult {
        canceled = false
        let beginDate = Date()

        var config = StableDiffusion.ZImageConfiguration(
            embeddingsURL: embeddingsURL,
            stepCount: stepCount,
            seed: seed
        )
        config.initialLatentData = initialNoiseData
        config.initialLatentShape = initialNoiseShape
        let debugRunDirectory = try makeDebugRunDirectory(seed: seed)
        config.debugEnabled = true
        config.debugOutputDirectory = debugRunDirectory
        config.debugSaveInitialLatent = true    
        config.debugSaveDitOutputEachStep = true
        config.debugSaveLatentAfterSchedulerEachStep = true
        config.debugSaveStageOutputs = true
        config.debugSkipVaeDecode = false
        // print("[ZImageDebug] Saving debug tensors to: \(debugRunDirectory.path)")

        let resourceURLs = transformerStageURLs + [vaeDecoderURL, embeddingsURL]
        let accessFlags = resourceURLs.map { $0.startAccessingSecurityScopedResource() }
        defer {
            for (url, granted) in zip(resourceURLs, accessFlags) where granted {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let images = try pipeline.generateImages(configuration: config) { zProgress in
            let sdProgress = StableDiffusionProgress(
                step: zProgress.step,
                stepCount: zProgress.stepCount
            )
            self.progressPublisher.value = sdProgress
            return !self.canceled
        }
        let interval = Date().timeIntervalSince(beginDate)
        let image = images.compactMap { $0 }.first
        return GenerationResult(
            image: image,
            lastSeed: seed,
            interval: interval,
            userCanceled: canceled,
            itsPerSecond: nil,
            initialNoiseData: nil,
            initialNoiseShape: nil
        )
    }

    func setCancelled() {
        canceled = true
    }

    private func makeDebugRunDirectory(seed: UInt32) throws -> URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        let root = appSupport.appending(path: "zimage-debug", directoryHint: .isDirectory)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)

        let timestampFormatter = ISO8601DateFormatter()
        timestampFormatter.formatOptions = [.withInternetDateTime, .withDashSeparatorInDate, .withColonSeparatorInTime]
        let timestamp = timestampFormatter.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let runDirectory = root.appending(path: "run_\(timestamp)_seed_\(seed)", directoryHint: .isDirectory)
        try fileManager.createDirectory(at: runDirectory, withIntermediateDirectories: true)
        return runDirectory
    }
}
