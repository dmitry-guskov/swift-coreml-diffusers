//
//  Pipeline.swift
//  Diffusion
//
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import Foundation
import CoreML
import CoreGraphics
import Combine

import StableDiffusion

protocol AppPipeline {
    var progressPublisher: CurrentValueSubject<StableDiffusionProgress?, Never> { get }

    func generate(
        prompt: String,
        negativePrompt: String,
        scheduler: StableDiffusionScheduler,
        numInferenceSteps stepCount: Int,
        seed: UInt32,
        numPreviews previewCount: Int,
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

struct StableDiffusionProgress {
    let step: Int
    let stepCount: Int
    var currentImages: [CGImage?]

    init(progress: StableDiffusionPipeline.Progress, previewIndices: [Bool]) {
        self.step = progress.step
        self.stepCount = progress.stepCount
        self.currentImages = [nil]

        if progress.step < previewIndices.count, previewIndices[progress.step] {
            self.currentImages = progress.currentImages
        }
    }

    init(step: Int, stepCount: Int) {
        self.step = step
        self.stepCount = stepCount
        self.currentImages = [nil]
    }
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

class Pipeline: AppPipeline {
    let pipeline: StableDiffusionPipelineProtocol
    let maxSeed: UInt32
    
    var isXL: Bool {
        if #available(macOS 14.0, iOS 17.0, *) {
            return (pipeline as? StableDiffusionXLPipeline) != nil
        }
        return false
    }

    var isSD3: Bool {
        if #available(macOS 14.0, iOS 17.0, *) {
            return (pipeline as? StableDiffusion3Pipeline) != nil
        }
        return false
    }

    var progress: StableDiffusionProgress? = nil {
        didSet {
            progressPublisher.value = progress
        }
    }
    lazy private(set) var progressPublisher: CurrentValueSubject<StableDiffusionProgress?, Never> = CurrentValueSubject(progress)
    
    private var canceled = false

    init(
        _ pipeline: StableDiffusionPipelineProtocol,
        maxSeed: UInt32 = UInt32.max
    ) {
        self.pipeline = pipeline
        self.maxSeed = maxSeed
    }
    
    func generate(
        prompt: String,
        negativePrompt: String = "",
        scheduler: StableDiffusionScheduler,
        numInferenceSteps stepCount: Int = 50,
        seed: UInt32 = 0,
        numPreviews previewCount: Int = 5,
        guidanceScale: Float = 7.5,
        disableSafety: Bool = false,
        startingImage: CGImage? = nil,
        strength: Float? = nil,
        initialNoiseData: Data? = nil,
        initialNoiseShape: [Int]? = nil,
        interpolationBaseNoiseData: Data? = nil,
        interpolationBaseNoiseShape: [Int]? = nil,
        interpolationSeed: UInt32? = nil,
        interpolationAmount: Float? = nil
    ) throws -> GenerationResult {
        let beginDate = Date()
        canceled = false
        let theSeed = seed > 0 ? seed : UInt32.random(in: 1...maxSeed)
        let sampleTimer = SampleTimer()
        sampleTimer.start()
        
        var config = StableDiffusionPipeline.Configuration(prompt: prompt)
        config.negativePrompt = negativePrompt
        config.stepCount = stepCount
        config.seed = theSeed
        config.guidanceScale = guidanceScale
        config.disableSafety = disableSafety
        config.schedulerType = scheduler.asStableDiffusionScheduler()
        config.useDenoisedIntermediates = true
        config.startingImage = startingImage
        if let strength {
            config.strength = max(0, min(1, strength))
        }
        config.initialNoiseData = initialNoiseData
        config.initialNoiseShape = initialNoiseShape
        config.interpolationBaseNoiseData = interpolationBaseNoiseData
        config.interpolationBaseNoiseShape = interpolationBaseNoiseShape
        config.interpolationSeed = interpolationSeed
        if let interpolationAmount {
            config.interpolationAmount = max(0, min(1, interpolationAmount))
        }
        if isXL {
            config.encoderScaleFactor = 0.13025
            config.decoderScaleFactor = 0.13025
            config.schedulerTimestepSpacing = .karras
        }

        if isSD3 {
            config.encoderScaleFactor = 1.5305
            config.decoderScaleFactor = 1.5305
            config.decoderShiftFactor = 0.0609
            config.schedulerTimestepShift = 3.0
        }

        // Evenly distribute previews based on inference steps
        let previewIndices = previewIndices(stepCount, previewCount)
        var capturedInitialNoiseData: Data? = nil
        var capturedInitialNoiseShape: [Int]? = nil

        let images = try pipeline.generateImages(configuration: config) { progress in
            if capturedInitialNoiseData == nil, let initialNoise = progress.initialLatentSamples.first {
                capturedInitialNoiseShape = initialNoise.shape
                capturedInitialNoiseData = initialNoise.scalars.withUnsafeBufferPointer { buffer in
                    Data(buffer: buffer)
                }
            }
            sampleTimer.stop()
            handleProgress(StableDiffusionProgress(progress: progress,
                                                   previewIndices: previewIndices),
                           sampleTimer: sampleTimer)
            if progress.stepCount != progress.step {
                sampleTimer.start()
            }
            return !canceled
        }
        let interval = Date().timeIntervalSince(beginDate)
        print("Got images: \(images) in \(interval)")
        
        // Unwrap the 1 image we asked for, nil means safety checker triggered
        let image = images.compactMap({ $0 }).first
        return GenerationResult(
            image: image,
            lastSeed: theSeed,
            interval: interval,
            userCanceled: canceled,
            itsPerSecond: 1.0 / sampleTimer.median,
            initialNoiseData: capturedInitialNoiseData,
            initialNoiseShape: capturedInitialNoiseShape
        )
    }

    func handleProgress(_ progress: StableDiffusionProgress, sampleTimer: SampleTimer) {
        self.progress = progress
    }
        
    func setCancelled() {
        canceled = true
    }
}
