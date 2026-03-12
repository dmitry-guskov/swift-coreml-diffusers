import CoreML
import Foundation

@available(iOS 17.0, macOS 14.0, *)
public extension ZImagePipeline {
    enum ResourceError: LocalizedError {
        case missingResource(path: String)

        public var errorDescription: String? {
            switch self {
            case .missingResource(let path):
                return "Missing Z-Image resource at path: \(path)"
            }
        }
    }

    static let transformerStageCount = 7

    static let transformerStageFileNames: [String] = (0..<transformerStageCount).map {
        "ZImageTurbo_TransformerBackbone_stage\($0).mlmodelc"
    }

    struct ResourceURLs {
        public let transformerStageURLs: [URL]
        public let vaeDecoderURL: URL

        public init(resourcesAt baseURL: URL) {
            let fm = FileManager.default
            transformerStageURLs = (0..<ZImagePipeline.transformerStageCount).map { i in
                let stem = "ZImageTurbo_TransformerBackbone_stage\(i)"
                let compiled = baseURL.appending(path: "\(stem).mlmodelc")
                if fm.fileExists(atPath: compiled.path) { return compiled }
                return baseURL.appending(path: "\(stem).mlpackage")
            }
            let compiledVAE = baseURL.appending(path: "VAEDecoder.mlmodelc")
            vaeDecoderURL = fm.fileExists(atPath: compiledVAE.path)
                ? compiledVAE
                : baseURL.appending(path: "VAEDecoder.mlpackage")
        }
    }

    init(
        transformerStagesAt stageURLs: [URL],
        vaeDecoderAt vaeDecoderURL: URL,
        configuration: MLModelConfiguration = .init(),
        reduceMemory: Bool = false
    ) throws {
        for url in stageURLs {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ResourceError.missingResource(path: url.path)
            }
        }
        guard FileManager.default.fileExists(atPath: vaeDecoderURL.path) else {
            throw ResourceError.missingResource(path: vaeDecoderURL.path)
        }

        let dit = Dit(stagesAt: stageURLs, configuration: configuration)

        let vaeConfig = MLModelConfiguration()
        vaeConfig.computeUnits = .all
        let vae = AutoencoderKLZImage(
            decoderAt: vaeDecoderURL,
            configuration: vaeConfig,
            scalingFactor: 0.3611,
            shiftFactor: 0.1159,
            latentChannels: 16
        )
        self.init(dit: dit, vae: vae, reduceMemory: reduceMemory)
    }

    init(
        resourcesAt baseURL: URL,
        configuration: MLModelConfiguration = .init(),
        reduceMemory: Bool = false
    ) throws {
        let urls = ResourceURLs(resourcesAt: baseURL)
        try self.init(
            transformerStagesAt: urls.transformerStageURLs,
            vaeDecoderAt: urls.vaeDecoderURL,
            configuration: configuration,
            reduceMemory: reduceMemory
        )
    }
}
