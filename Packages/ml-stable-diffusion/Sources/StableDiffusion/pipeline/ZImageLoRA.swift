import CoreML
import Foundation

@available(iOS 18.0, macOS 14.0, *)
enum ZImageLoRAError: LocalizedError {
    case invalidSafetensorsHeader(URL)
    case unsupportedTensorType(name: String, dtype: String)
    case missingTensor(name: String)
    case invalidTensorShape(name: String, expected: [Int], actual: [Int])
    case invalidTensorOffsets(name: String)
    case invalidStageIndex(Int)

    var errorDescription: String? {
        switch self {
        case .invalidSafetensorsHeader(let url):
            return "Invalid safetensors header in \(url.lastPathComponent)."
        case .unsupportedTensorType(let name, let dtype):
            return "Unsupported safetensors dtype \(dtype) for tensor \(name). Expected F32."
        case .missingTensor(let name):
            return "Missing LoRA tensor \(name)."
        case .invalidTensorShape(let name, let expected, let actual):
            return "Invalid shape for LoRA tensor \(name). Expected \(expected), got \(actual)."
        case .invalidTensorOffsets(let name):
            return "Invalid data offsets for LoRA tensor \(name)."
        case .invalidStageIndex(let stageIndex):
            return "Invalid Z-Image LoRA stage index \(stageIndex)."
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
final class ZImageLoRAInputProvider {
    private static let blocksPerStage = 1
    private static let totalBlockStages = 30
    private static let rank = 32
    private static let nHeads = 30
    private static let headDim = 128

    private struct Target {
        let path: String
        let aShape: [Int]
        let bShape: [Int]
        let permuteB: Bool
    }

    private struct TensorDescriptor {
        let dtype: String
        let shape: [Int]
        let start: Int
        let end: Int
    }

    private static let targets: [Target] = [
        Target(path: "adaLN_modulation.0", aShape: [rank, 256], bShape: [15360, rank], permuteB: false),
        Target(path: "attention.to_k", aShape: [rank, 3840], bShape: [3840, rank], permuteB: true),
        Target(path: "attention.to_out.0", aShape: [rank, 3840], bShape: [3840, rank], permuteB: false),
        Target(path: "attention.to_q", aShape: [rank, 3840], bShape: [3840, rank], permuteB: true),
        Target(path: "attention.to_v", aShape: [rank, 3840], bShape: [3840, rank], permuteB: false),
        Target(path: "feed_forward.w1", aShape: [rank, 3840], bShape: [10240, rank], permuteB: false),
        Target(path: "feed_forward.w2", aShape: [rank, 10240], bShape: [3840, rank], permuteB: false),
        Target(path: "feed_forward.w3", aShape: [rank, 3840], bShape: [10240, rank], permuteB: false),
    ]

    private static let stageVectorLength: Int = {
        let perLayer = targets.reduce(0) { partialResult, target in
            partialResult + elementCount(for: target.aShape) + elementCount(for: target.bShape)
        }
        return blocksPerStage * perLayer
    }()

    private let archive: Archive?
    private let zeroVector: MLMultiArray
    private let zeroScale: MLMultiArray
    private let activeScale: MLMultiArray

    init(loraURL: URL?, scale: Float32) throws {
        self.archive = try loraURL.map(Archive.init(contentsOf:))
        self.zeroVector = try Self.makeZeroVector()
        self.zeroScale = try Self.makeScalarArray(0)
        self.activeScale = try Self.makeScalarArray(scale)
    }

    func featureValues(for stageIndex: Int) throws -> (vector: MLFeatureValue, scale: MLFeatureValue) {
        if stageIndex <= 1 || archive == nil {
            return (MLFeatureValue(multiArray: zeroVector), MLFeatureValue(multiArray: zeroScale))
        }
        guard let archive else {
            return (MLFeatureValue(multiArray: zeroVector), MLFeatureValue(multiArray: zeroScale))
        }
        let vector = try archive.packStageVector(stageIndex: stageIndex)
        return (MLFeatureValue(multiArray: vector), MLFeatureValue(multiArray: activeScale))
    }

    private static func makeZeroVector() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [NSNumber(value: stageVectorLength)], dataType: .float32)
        let count = stageVectorLength
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
        for index in 0..<count {
            ptr[index] = 0
        }
        return array
    }

    private static func makeScalarArray(_ value: Float32) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [NSNumber(value: 1)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: 1)
        ptr[0] = value
        return array
    }

    private static func elementCount(for shape: [Int]) -> Int {
        shape.reduce(1, *)
    }

    private final class Archive {
        private let url: URL
        private let data: Data
        private let tensorDataOffset: Int
        private let descriptors: [String: TensorDescriptor]

        init(contentsOf url: URL) throws {
            self.url = url
            self.data = try Data(contentsOf: url, options: [.mappedIfSafe])

            guard data.count >= 8 else {
                throw ZImageLoRAError.invalidSafetensorsHeader(url)
            }

            let headerLength = data.prefix(8).enumerated().reduce(UInt64(0)) { partialResult, item in
                partialResult | (UInt64(item.element) << (item.offset * 8))
            }
            let headerStart = 8
            let headerEnd = headerStart + Int(headerLength)
            guard headerEnd <= data.count else {
                throw ZImageLoRAError.invalidSafetensorsHeader(url)
            }

            let headerData = data.subdata(in: headerStart..<headerEnd)
            let json = try JSONSerialization.jsonObject(with: headerData)
            guard let dict = json as? [String: Any] else {
                throw ZImageLoRAError.invalidSafetensorsHeader(url)
            }

            self.tensorDataOffset = headerEnd
            var parsed: [String: TensorDescriptor] = [:]
            for (name, rawValue) in dict {
                guard name != "__metadata__" else { continue }
                guard
                    let entry = rawValue as? [String: Any],
                    let dtype = entry["dtype"] as? String,
                    let shapeRaw = entry["shape"] as? [NSNumber],
                    let offsetsRaw = entry["data_offsets"] as? [NSNumber],
                    offsetsRaw.count == 2
                else {
                    throw ZImageLoRAError.invalidSafetensorsHeader(url)
                }

                parsed[name] = TensorDescriptor(
                    dtype: dtype,
                    shape: shapeRaw.map(\.intValue),
                    start: offsetsRaw[0].intValue,
                    end: offsetsRaw[1].intValue
                )
            }
            self.descriptors = parsed
        }

        func packStageVector(stageIndex: Int) throws -> MLMultiArray {
            let firstBlockStage = 2
            let lastBlockStage = firstBlockStage + ZImageLoRAInputProvider.totalBlockStages - 1
            guard (firstBlockStage...lastBlockStage).contains(stageIndex) else {
                throw ZImageLoRAError.invalidStageIndex(stageIndex)
            }

            let vector = try MLMultiArray(
                shape: [NSNumber(value: ZImageLoRAInputProvider.stageVectorLength)],
                dataType: .float32
            )
            let elementCount = ZImageLoRAInputProvider.stageVectorLength
            let destination = vector.dataPointer.bindMemory(to: Float32.self, capacity: elementCount)

            let layerStart = (stageIndex - firstBlockStage) * ZImageLoRAInputProvider.blocksPerStage
            let layerEnd = layerStart + ZImageLoRAInputProvider.blocksPerStage
            var cursor = 0

            try data.withUnsafeBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else {
                    throw ZImageLoRAError.invalidSafetensorsHeader(url)
                }

                for layerIndex in layerStart..<layerEnd {
                    for target in ZImageLoRAInputProvider.targets {
                        let aName = "diffusion_model.layers.\(layerIndex).\(target.path).lora_A.weight"
                        let bName = "diffusion_model.layers.\(layerIndex).\(target.path).lora_B.weight"

                        cursor += try copyTensor(named: aName, expectedShape: target.aShape, from: baseAddress, into: destination.advanced(by: cursor))
                        let bCount = try copyTensor(named: bName, expectedShape: target.bShape, from: baseAddress, into: destination.advanced(by: cursor))
                        if target.permuteB {
                            Self.permuteInterleavedToHalfRotation(
                                destination.advanced(by: cursor),
                                rows: target.bShape[0],
                                cols: target.bShape[1],
                                nHeads: ZImageLoRAInputProvider.nHeads,
                                headDim: ZImageLoRAInputProvider.headDim
                            )
                        }
                        cursor += bCount
                    }
                }
            }

            return vector
        }

        /// Permute B-matrix rows from interleaved RoPE convention to half-rotation.
        ///
        /// Interleaved: [r0, i0, r1, i1, …, r63, i63] per head
        /// Half-rotation: [r0, r1, …, r63, i0, i1, …, i63] per head
        ///
        /// For each head's `headDim` consecutive rows, even-indexed rows (0,2,4,…)
        /// move to the first half, odd-indexed rows (1,3,5,…) to the second half.
        /// The matrix is row-major with `cols` columns per row.
        private static func permuteInterleavedToHalfRotation(
            _ buffer: UnsafeMutablePointer<Float32>,
            rows: Int,
            cols: Int,
            nHeads: Int,
            headDim: Int
        ) {
            assert(rows == nHeads * headDim)
            let halfDim = headDim / 2
            let rowBytes = cols * MemoryLayout<Float32>.size
            let tmp = UnsafeMutablePointer<Float32>.allocate(capacity: headDim * cols)
            defer { tmp.deallocate() }

            for h in 0..<nHeads {
                let headOffset = h * headDim
                let src = buffer.advanced(by: headOffset * cols)
                memcpy(tmp, src, headDim * rowBytes)

                for i in 0..<halfDim {
                    let evenSrc = tmp.advanced(by: (2 * i) * cols)
                    let oddSrc = tmp.advanced(by: (2 * i + 1) * cols)
                    let evenDst = src.advanced(by: i * cols)
                    let oddDst = src.advanced(by: (halfDim + i) * cols)
                    memcpy(evenDst, evenSrc, rowBytes)
                    memcpy(oddDst, oddSrc, rowBytes)
                }
            }
        }

        private func copyTensor(
            named name: String,
            expectedShape: [Int],
            from baseAddress: UnsafeRawPointer,
            into destination: UnsafeMutablePointer<Float32>
        ) throws -> Int {
            guard let descriptor = descriptors[name] else {
                throw ZImageLoRAError.missingTensor(name: name)
            }
            guard descriptor.dtype == "F32" else {
                throw ZImageLoRAError.unsupportedTensorType(name: name, dtype: descriptor.dtype)
            }
            guard descriptor.shape == expectedShape else {
                throw ZImageLoRAError.invalidTensorShape(name: name, expected: expectedShape, actual: descriptor.shape)
            }

            let start = tensorDataOffset + descriptor.start
            let end = tensorDataOffset + descriptor.end
            let expectedByteCount = ZImageLoRAInputProvider.elementCount(for: expectedShape) * MemoryLayout<Float32>.size
            guard
                descriptor.start >= 0,
                descriptor.end >= descriptor.start,
                end <= data.count,
                (descriptor.end - descriptor.start) == expectedByteCount
            else {
                throw ZImageLoRAError.invalidTensorOffsets(name: name)
            }

            let source = baseAddress.advanced(by: start)
            UnsafeMutableRawPointer(destination).copyMemory(from: source, byteCount: expectedByteCount)
            return expectedByteCount / MemoryLayout<Float32>.size
        }
    }
}
