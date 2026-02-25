//
//  DecoderZImage.swift  →  AutoencoderKLZImage
//  stable-diffusion
//
//  Created by Dmitry Guskov on 10.02.2026.
//

import Foundation
import CoreML
import CoreGraphics
import Accelerate

// MARK: - AutoencoderKLZImage

/// Unified VAE (encoder + decoder) for Z-Image, mirroring the Python `AutoencoderKL`.
///
/// The CoreML export should produce **two separate** compiled models:
///   - **Encoder**: `image [1,3,H,W]` → `latent_dist [1, 2*C, h, w]`
///     (Encoder + quant_conv baked in; outputs mean ‖ logvar along channel axis)
///   - **Decoder**: `latent [1, C, h, w]` → `image [1,3,H,W]`
///     (post_quant_conv + Decoder baked in; output in [-1, 1])
///
/// Either model is **optional** so the caller can load only what it needs.
/// The VAE **owns** its `scalingFactor` and `shiftFactor` — the pipeline
/// should not need to know about them.
///
/// ### Python equivalence
/// ```python
/// # Encode
/// posterior = vae.encode(image).latent_dist   # quant_conv inside
/// latent = posterior.sample()
/// latent = (latent - shift_factor) * scaling_factor
///
/// # Decode
/// latent = (latent / scaling_factor) + shift_factor
/// image  = vae.decode(latent)                 # post_quant_conv inside
/// image  = (image / 2 + 0.5).clamp(0, 1)
/// ```
@available(iOS 17.0, macOS 14.0, *)
public struct AutoencoderKLZImage: ResourceManaging {

    // MARK: - Errors

    public enum Error: String, Swift.Error {
        case encoderNotLoaded
        case decoderNotLoaded
        case inputShapeMismatch
    }

    // MARK: - Config (owned by the VAE, not the pipeline)

    /// Multiplier applied after encoding / divisor before decoding.
    /// Default `0.18215` from `model.py → DEFAULT_VAE_SCALING_FACTOR`.
    public let scalingFactor: Float32

    /// Additive shift applied after encoding / before decoding.
    /// Default `0.0`; read from `vae/config.json "shift_factor"`.
    public let shiftFactor: Float32

    /// Number of latent channels (4 for Z-Image).
    public let latentChannels: Int

    // MARK: - Models

    /// Encoder CoreML model (optional — not needed for text-to-image)
    private var encoderModel: ManagedMLModel?

    /// Decoder CoreML model (optional — not needed for encode-only workflows)
    private var decoderModel: ManagedMLModel?

    // MARK: - Initializers

    /// Full initializer — supply whichever models you have.
    ///
    /// - Parameters:
    ///   - encoderURL: Compiled `.mlmodelc` for the VAE encoder (nil if unused)
    ///   - decoderURL: Compiled `.mlmodelc` for the VAE decoder (nil if unused)
    ///   - configuration: Core ML configuration (compute units, etc.)
    ///   - scalingFactor: VAE scaling factor (default 0.18215)
    ///   - shiftFactor: VAE shift factor (default 0.0)
    ///   - latentChannels: Number of latent channels (default 4)
    public init(
        encoderURL: URL? = nil,
        decoderURL: URL? = nil,
        configuration: MLModelConfiguration,
        scalingFactor: Float32 = 0.18215,
        shiftFactor: Float32 = 0.0,
        latentChannels: Int = 4
    ) {
        self.encoderModel = encoderURL.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.decoderModel = decoderURL.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.scalingFactor = scalingFactor
        self.shiftFactor = shiftFactor
        self.latentChannels = latentChannels
    }

    /// Convenience: decoder-only (text-to-image pipeline)
    public init(
        decoderAt url: URL,
        configuration: MLModelConfiguration,
        scalingFactor: Float32 = 0.18215,
        shiftFactor: Float32 = 0.0,
        latentChannels: Int = 4
    ) {
        self.init(
            encoderURL: nil,
            decoderURL: url,
            configuration: configuration,
            scalingFactor: scalingFactor,
            shiftFactor: shiftFactor,
            latentChannels: latentChannels
        )
    }

    /// Convenience: both encoder + decoder (img2img, inpainting, future workflows)
    public init(
        encoderAt encoderURL: URL,
        decoderAt decoderURL: URL,
        configuration: MLModelConfiguration,
        scalingFactor: Float32 = 0.18215,
        shiftFactor: Float32 = 0.0,
        latentChannels: Int = 4
    ) {
        self.init(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            configuration: configuration,
            scalingFactor: scalingFactor,
            shiftFactor: shiftFactor,
            latentChannels: latentChannels
        )
    }

    // MARK: - ResourceManaging

    public func loadResources() throws {
        try encoderModel?.loadResources()
        try decoderModel?.loadResources()
    }

    public func unloadResources() {
        encoderModel?.unloadResources()
        decoderModel?.unloadResources()
    }

    /// Load encoder only (img2img preparation)
    public func loadEncoder() throws {
        try encoderModel?.loadResources()
    }

    /// Load decoder only (standard decode path)
    public func loadDecoder() throws {
        try decoderModel?.loadResources()
    }

    /// Unload encoder only (free memory after encoding)
    public func unloadEncoder() {
        encoderModel?.unloadResources()
    }

    /// Unload decoder only
    public func unloadDecoder() {
        decoderModel?.unloadResources()
    }

    public func prewarmResources() throws {
        if let enc = encoderModel {
            try autoreleasepool {
                try enc.loadResources()
                enc.unloadResources()
            }
        }
        if let dec = decoderModel {
            try autoreleasepool {
                try dec.loadResources()
                dec.unloadResources()
            }
        }
    }

    // MARK: - Encode

    /// Encode a CGImage into a scaled latent sample.
    ///
    /// Runs the VAE encoder (with `quant_conv` baked in), then performs
    /// DiagonalGaussianDistribution sampling, then applies:
    /// ```
    /// latent = (sample - shift_factor) * scaling_factor
    /// ```
    ///
    /// - Parameters:
    ///   - image: Input RGB image
    ///   - random: Mutable random source for reparameterisation trick
    /// - Returns: Scaled latent `[1, C, h, w]`
    public func encode(
        _ image: CGImage,
        random: inout RandomSource
    ) throws -> MLShapedArray<Float32> {
        guard let encoder = encoderModel else {
            throw Error.encoderNotLoaded
        }

        // Convert CGImage to planar RGB [-1, 1]
        let imageData = try image.planarRGBShapedArray(minValue: -1.0, maxValue: 1.0)

        guard imageData.shape == encoderInputShape else {
            throw Error.inputShapeMismatch
        }

        let dict = [encoderInputName: MLMultiArray(imageData)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)

        let result = try encoder.perform { model in
            try model.prediction(from: input)
        }

        let outputName = result.featureNames.first!
        let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
        let output = MLShapedArray<Float32>(converting: outputValue)

        // ── DiagonalGaussianDistribution ──
        // Encoder output is [1, 2*C, h, w] — first C channels = mean, next C = logvar
        let c = latentChannels
        let mean   = output[0][0..<c]
        let logvar = MLShapedArray<Float32>(
            scalars: output[0][c..<(2 * c)].scalars.map { min(max($0, -30.0), 20.0) },
            shape: mean.shape
        )
        let std = MLShapedArray<Float32>(
            scalars: logvar.scalars.map { exp(0.5 * $0) },
            shape: logvar.shape
        )

        // Reparameterisation trick: sample = mean + std * ε
        let sample = MLShapedArray<Float32>(
            scalars: zip(mean.scalars, std.scalars).map { m, s in
                Float32(random.nextNormal(mean: Double(m), stdev: Double(s)))
            },
            shape: mean.shape
        )

        // Apply VAE scaling: latent = (sample - shift_factor) * scaling_factor
        let latent = MLShapedArray<Float32>(
            scalars: sample.scalars.map { ($0 - shiftFactor) * scalingFactor },
            shape: [1] + sample.shape   // [1, C, h, w]
        )

        return latent
    }

    // MARK: - Decode

    /// Decode latent samples into images.
    ///
    /// Applies inverse VAE scaling then runs the decoder model:
    /// ```
    /// z = (latent / scaling_factor) + shift_factor
    /// image = decoder(z)                         // output in [-1, 1]
    /// image = (image / 2 + 0.5).clamp(0, 1)
    /// ```
    ///
    /// - Parameter latents: Scaled latent samples, each `[1, C, h, w]`
    /// - Returns: Decoded `CGImage`s (nil if conversion fails for a sample)
    public func decode(
        _ latents: [MLShapedArray<Float32>]
    ) throws -> [CGImage?] {
        guard let decoder = decoderModel else {
            throw Error.decoderNotLoaded
        }

        // Prepare inputs: inverse of encode scaling
        // Python: latents = (latents / scaling_factor) + shift_factor
        let inputs: [MLFeatureProvider] = try latents.map { sample in
            let unscaled = MLShapedArray<Float32>(
                scalars: sample.scalars.map { ($0 / scalingFactor) + shiftFactor },
                shape: sample.shape
            )
            let dict = [decoderInputName: MLMultiArray(unscaled)]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        let results = try decoder.perform { model in
            try model.predictions(fromBatch: batch)
        }

        // Convert each result to CGImage
        let images: [CGImage?] = (0..<results.count).map { i in
            let result = results.features(at: i)
            let outputName = result.featureNames.first!
            let output = result.featureValue(for: outputName)!.multiArrayValue!

            // Try the existing SD helper (expects specific shaped-array layout)
            if let image = try? CGImage.fromShapedArray(MLShapedArray<Float32>(converting: output)) {
                return image
            }

            // Fallback: manual [-1, 1] → [0, 255] → CGImage
            return cgImageFromModelOutput(output)
        }

        return images
    }

    // MARK: - Model metadata helpers

    /// Auto-detect the encoder model's input key name
    var encoderInputName: String {
        try! encoderModel!.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.key
        }
    }

    /// Expected input shape for the encoder (e.g. [1, 3, H, W])
    var encoderInputShape: [Int] {
        try! encoderModel!.perform { model in
            let desc = model.modelDescription.inputDescriptionsByName.first!.value
            return desc.multiArrayConstraint!.shape.map { $0.intValue }
        }
    }

    /// Auto-detect the decoder model's input key name
    var decoderInputName: String {
        try! decoderModel!.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.key
        }
    }

    // MARK: - Image conversion fallback

    /// Convert VAE decoder output from [-1, 1] range to a CGImage.
    ///
    /// Expected output shape: `[1, 3, H, W]` or `[3, H, W]` in CHW planar format.
    /// Pixel transform: `pixel = clamp((value / 2 + 0.5), 0, 1) * 255`
    private func cgImageFromModelOutput(_ multiArray: MLMultiArray) -> CGImage? {
        let shape = (0..<multiArray.shape.count).map { multiArray.shape[$0].intValue }
        let channels: Int
        let height: Int
        let width: Int

        if shape.count == 4 {
            channels = shape[1]; height = shape[2]; width = shape[3]
        } else if shape.count == 3 {
            channels = shape[0]; height = shape[1]; width = shape[2]
        } else {
            return nil
        }

        guard channels == 3 else { return nil }

        let pixelCount = height * width
        var pixelData = [UInt8](repeating: 0, count: pixelCount * 4)

        let ptr = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: multiArray.count)

        for y in 0..<height {
            for x in 0..<width {
                let px = y * width + x
                for c in 0..<3 {
                    let value = ptr[c * pixelCount + px]
                    let norm = min(max(value / 2.0 + 0.5, 0.0), 1.0)
                    pixelData[px * 4 + c] = UInt8(norm * 255.0)
                }
                pixelData[px * 4 + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData) else { return nil }

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}

// MARK: - Legacy type alias

/// Backward-compatible alias so existing pipeline code compiles
/// while we migrate from the old name.
@available(iOS 17.0, macOS 14.0, *)
public typealias DecoderZImage = AutoencoderKLZImage
