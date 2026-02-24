import CoreML
import CoreGraphics
import Foundation
import StableDiffusion

@available(iOS 17.0, macOS 14.0, *)
struct ZImageBootstrapConfig {
    let transformerURL: URL
    let vaeDecoderURL: URL
    let embeddingsURL: URL

    init(
        transformerURL: URL,
        vaeDecoderURL: URL,
        embeddingsURL: URL
    ) {
        self.transformerURL = transformerURL
        self.vaeDecoderURL = vaeDecoderURL
        self.embeddingsURL = embeddingsURL
    }
}

@available(iOS 17.0, macOS 14.0, *)
enum ZImagePipelineLoaderError: LocalizedError {
    case missingTransformer(path: String)
    case missingVaeDecoder(path: String)
    case missingEmbeddings(path: String)

    var errorDescription: String? {
        switch self {
        case .missingTransformer(let path):
            return "Missing Transformer model at path: \(path)"
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
        computeUnits: ComputeUnits = .cpuAndNeuralEngine
    ) {
        self.config = config
        self.computeUnits = computeUnits
    }

    private func validateResources() throws {
        guard FileManager.default.fileExists(atPath: config.transformerURL.path) else {
            throw ZImagePipelineLoaderError.missingTransformer(path: config.transformerURL.path)
        }
        guard FileManager.default.fileExists(atPath: config.vaeDecoderURL.path) else {
            throw ZImagePipelineLoaderError.missingVaeDecoder(path: config.vaeDecoderURL.path)
        }
        guard FileManager.default.fileExists(atPath: config.embeddingsURL.path) else {
            throw ZImagePipelineLoaderError.missingEmbeddings(path: config.embeddingsURL.path)
        }
    }

    private func withResourceAccess<T>(_ body: () throws -> T) throws -> T {
        let urls = [config.transformerURL, config.vaeDecoderURL, config.embeddingsURL]
        let accessFlags = urls.map { $0.startAccessingSecurityScopedResource() }
        defer {
            for (index, granted) in accessFlags.enumerated().reversed() where granted {
                urls[index].stopAccessingSecurityScopedResource()
            }
        }
        return try body()
    }

    private func loadUnchecked() throws -> ZImagePipeline {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        return try ZImagePipeline(
            transformerAt: config.transformerURL,
            vaeDecoderAt: config.vaeDecoderURL,
            configuration: mlConfig,
            reduceMemory: true
        )
    }

    func load() throws -> ZImagePipeline {
        try withResourceAccess {
            try validateResources()
            return try loadUnchecked()
        }
    }

    func loadAppPipeline(
        runSmokeTest: Bool = true,
        smokeSteps: Int = 4,
        smokeSeed: UInt32 = 42
    ) throws -> AppPipeline {
        if runSmokeTest {
            _ = try self.runSmokeTest(stepCount: smokeSteps, seed: smokeSeed)
        }
        let pipeline = try load()
        return ZImageAppPipeline(
            pipeline: pipeline,
            transformerURL: config.transformerURL,
            vaeDecoderURL: config.vaeDecoderURL,
            embeddingsURL: config.embeddingsURL
        )
    }

    @discardableResult
    func runSmokeTest(stepCount: Int = 4, seed: UInt32 = 42) throws -> CGImage? {
        try withResourceAccess {
            try validateResources()
            try logModelContract()
            print("[ZImageSmoke] embeddings_path=\(config.embeddingsURL.path)")
            print("[ZImageSmoke] expected_latents_shape=[1,16,64,64] expected_cap_feats_shape=[1,77,2560]")

            var pipeline = try loadUnchecked()
            try pipeline.loadResources()
            defer { pipeline.unloadResources() }

            var finalLatentStats: (min: Float32, max: Float32)?
            let generationConfig = ZImageConfiguration(
                embeddingsURL: config.embeddingsURL,
                stepCount: stepCount,
                seed: seed
            )
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

            if let stats = finalLatentStats {
                print("[ZImageSmoke] final_latent_min=\(stats.min) final_latent_max=\(stats.max)")
            }
            print("[ZImageSmoke] output_images=\(images.count)")
            return images.first ?? nil
        }
    }

    private func logModelContract() throws {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        let model = try MLModel(contentsOf: config.transformerURL, configuration: mlConfig)
        let inputNames = model.modelDescription.inputDescriptionsByName.keys.sorted()
        print("[ZImageSmoke] model_input_names=\(inputNames)")
    }
}
