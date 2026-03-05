import CoreML
import CoreGraphics
import Foundation
import StableDiffusion

@available(iOS 18.0, macOS 14.0, *)
struct ZImageBootstrapConfig {
    let transformerStageURLs: [URL]
    let vaeDecoderURL: URL
    let embeddingsURL: URL

    init(
        transformerStageURLs: [URL],
        vaeDecoderURL: URL,
        embeddingsURL: URL
    ) {
        self.transformerStageURLs = transformerStageURLs
        self.vaeDecoderURL = vaeDecoderURL
        self.embeddingsURL = embeddingsURL
    }
}

@available(iOS 18.0, macOS 14.0, *)
enum ZImagePipelineLoaderError: LocalizedError {
    case missingTransformerStage(index: Int, path: String)
    case missingVaeDecoder(path: String)
    case missingEmbeddings(path: String)

    var errorDescription: String? {
        switch self {
        case .missingTransformerStage(let index, let path):
            return "Missing Transformer stage \(index) at path: \(path)"
        case .missingVaeDecoder(let path):
            return "Missing VAE decoder model at path: \(path)"
        case .missingEmbeddings(let path):
            return "Missing embeddings tensor file at path: \(path)"
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
final class ZImagePipelineLoader {
    private let config: ZImageBootstrapConfig
    private let computeUnits: ComputeUnits

    init(
        config: ZImageBootstrapConfig,
        computeUnits: ComputeUnits = .cpuOnly
    ) {
        self.config = config
        self.computeUnits = computeUnits
    }

    private func logResolvedResources() {
        let fm = FileManager.default
        print("[PipelineLoader] ===== Resolved resource manifest =====")
        print("[PipelineLoader] computeUnits = \(computeUnits)")
        for (i, url) in config.transformerStageURLs.enumerated() {
            let exists = fm.fileExists(atPath: url.path)
            print("[PipelineLoader]   stage[\(i)] = \(url.path)  (exists: \(exists))")
        }
        let vaeExists = fm.fileExists(atPath: config.vaeDecoderURL.path)
        print("[PipelineLoader]   vaeDecoder = \(config.vaeDecoderURL.path)  (exists: \(vaeExists))")
        let embExists = fm.fileExists(atPath: config.embeddingsURL.path)
        print("[PipelineLoader]   embeddings = \(config.embeddingsURL.path)  (exists: \(embExists))")
        print("[PipelineLoader] ===== End resource manifest =====")
    }

    private func validateResources() throws {
        logResolvedResources()
        for (i, url) in config.transformerStageURLs.enumerated() {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ZImagePipelineLoaderError.missingTransformerStage(index: i, path: url.path)
            }
        }
        guard FileManager.default.fileExists(atPath: config.vaeDecoderURL.path) else {
            throw ZImagePipelineLoaderError.missingVaeDecoder(path: config.vaeDecoderURL.path)
        }
        guard FileManager.default.fileExists(atPath: config.embeddingsURL.path) else {
            throw ZImagePipelineLoaderError.missingEmbeddings(path: config.embeddingsURL.path)
        }
    }

    private func withResourceAccess<T>(_ body: () throws -> T) throws -> T {
        let urls = config.transformerStageURLs + [config.vaeDecoderURL, config.embeddingsURL]
        let accessFlags = urls.map { $0.startAccessingSecurityScopedResource() }
        defer {
            for (index, granted) in accessFlags.enumerated().reversed() where granted {
                urls[index].stopAccessingSecurityScopedResource()
            }
        }
        return try body()
    }

    private let debugSingleStageMode = false // flip to true for single-model debug

    private func loadUnchecked() throws -> ZImagePipeline {
        print("[PipelineLoader] loadUnchecked.start")
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        print("[PipelineLoader] computeUnits = \(mlConfig.computeUnits) (requested: \(computeUnits))")

        let stages: [URL]
        if debugSingleStageMode {
            stages = [config.transformerStageURLs.first!]
            print("[PipelineLoader] DEBUG: single-model mode — \(stages[0].lastPathComponent)")
        } else {
            stages = config.transformerStageURLs
            print("[PipelineLoader] Loading all \(stages.count) stages")
        }
        for (i, url) in stages.enumerated() {
            print("[PipelineLoader]   stage[\(i)] = \(url.lastPathComponent)")
        }

        let pipeline = try ZImagePipeline(
            transformerStagesAt: stages,
            vaeDecoderAt: config.vaeDecoderURL,
            configuration: mlConfig,
            reduceMemory: !debugSingleStageMode
        )
        print("[PipelineLoader] loadUnchecked.pipelineCreated")
        return pipeline
    }

    func load() throws -> ZImagePipeline {
        print("[PipelineLoader] load.start")
        return try withResourceAccess {
            try validateResources()
            let pipeline = try loadUnchecked()
            print("[PipelineLoader] load.complete")
            return pipeline
        }
    }

    func loadAppPipeline(
        runSmokeTest: Bool = true,
        smokeSteps: Int = 4,
        smokeSeed: UInt32 = 42
    ) throws -> AppPipeline {
        print("[PipelineLoader] loadAppPipeline.start")
        if runSmokeTest {
            print("[PipelineLoader] loadAppPipeline.beforeSmokeTest")
            _ = try self.runSmokeTest(stepCount: smokeSteps, seed: smokeSeed)
            print("[PipelineLoader] loadAppPipeline.afterSmokeTest")
        }
        print("[PipelineLoader] loadAppPipeline.beforeLoad")
        let pipeline = try load()
        print("[PipelineLoader] loadAppPipeline.afterLoad")
        let appPipeline = ZImageAppPipeline(
            pipeline: pipeline,
            transformerStageURLs: config.transformerStageURLs,
            vaeDecoderURL: config.vaeDecoderURL,
            embeddingsURL: config.embeddingsURL
        )
        print("[PipelineLoader] loadAppPipeline.complete")
        return appPipeline
    }

    @discardableResult
    func runSmokeTest(stepCount: Int = 4, seed: UInt32 = 42) throws -> CGImage? {
        print("[SmokeTest] start")
        return try withResourceAccess {
            try validateResources()
            try logModelContract()
            print("[ZImageSmoke] embeddings_path=\(config.embeddingsURL.path)")
            print("[ZImageSmoke] expected_latents_shape=[1,16,64,64] expected_cap_feats_shape=[1,77,2560]")

            print("[SmokeTest] beforePipelineCreate")
            var pipeline = try loadUnchecked()
            print("[SmokeTest] afterPipelineCreate")
            
            print("[SmokeTest] beforeLoadResources")
            try pipeline.loadResources()
            print("[SmokeTest] afterLoadResources")
            defer {
                print("[SmokeTest] beforeUnloadResources")
                pipeline.unloadResources()
                print("[SmokeTest] afterUnloadResources")
            }

            var finalLatentStats: (min: Float32, max: Float32)?
            let generationConfig = ZImageConfiguration(
                embeddingsURL: config.embeddingsURL,
                stepCount: stepCount,
                seed: seed
            )
            
            print("[SmokeTest] beforeGenerate")
            let images = try pipeline.generateImages(configuration: generationConfig) { progress in
                let shape = progress.currentLatentSample.shape
                let scalars = progress.currentLatentSample.scalars
                if let minValue = scalars.min(), let maxValue = scalars.max() {
                    finalLatentStats = (minValue, maxValue)
                    print("[ZImageSmoke] step=\(progress.step + 1)/\(progress.stepCount) latent_shape=\(shape) latent_min=\(minValue) latent_max=\(maxValue)")
                } else {
                    print("[ZImageSmoke] step=\(progress.step + 1)/\(progress.stepCount) latent_shape=\(shape)")
                }
                return true
            }
            print("[SmokeTest] afterGenerate")

            if let stats = finalLatentStats {
                print("[ZImageSmoke] final_latent_min=\(stats.min) final_latent_max=\(stats.max)")
            }
            print("[ZImageSmoke] output_images=\(images.count)")
            print("[SmokeTest] complete")
            return images.first ?? nil
        }
    }

    private func logModelContract() throws {
        guard let firstStageURL = config.transformerStageURLs.first else { return }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        let model = try MLModel(contentsOf: firstStageURL, configuration: mlConfig)
        let inputNames = model.modelDescription.inputDescriptionsByName.keys.sorted()
        print("[ZImageSmoke] stage0_input_names=\(inputNames)")
    }
}
