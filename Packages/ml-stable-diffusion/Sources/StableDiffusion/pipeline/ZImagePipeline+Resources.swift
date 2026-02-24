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

    struct ResourceURLs {
        public let transformerURL: URL
        public let vaeDecoderURL: URL

        public init(resourcesAt baseURL: URL) {
            transformerURL = baseURL.appending(path: "ZImageTurbo_TransformerBackbone.mlmodelc")
            vaeDecoderURL = baseURL.appending(path: "VAEDecoder.mlmodelc")
        }
    }

    init(
        transformerAt transformerURL: URL,
        vaeDecoderAt vaeDecoderURL: URL,
        configuration: MLModelConfiguration = .init(),
        reduceMemory: Bool = false
    ) throws {
        guard FileManager.default.fileExists(atPath: transformerURL.path) else {
            throw ResourceError.missingResource(path: transformerURL.path)
        }
        guard FileManager.default.fileExists(atPath: vaeDecoderURL.path) else {
            throw ResourceError.missingResource(path: vaeDecoderURL.path)
        }

        let dit = Dit(modelAt: transformerURL, configuration: configuration)
        let vae = AutoencoderKLZImage(
            decoderAt: vaeDecoderURL,
            configuration: configuration,
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
            transformerAt: urls.transformerURL,
            vaeDecoderAt: urls.vaeDecoderURL,
            configuration: configuration,
            reduceMemory: reduceMemory
        )
    }
}
