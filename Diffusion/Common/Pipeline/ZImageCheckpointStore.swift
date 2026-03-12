import CoreML
import Foundation

enum ZImageCheckpointSourceType: String, Codable {
    case bundle
    case imported
}

struct ZImageCheckpointSet: Identifiable, Codable, Hashable {
    let id: String
    var displayName: String
    var sourceType: ZImageCheckpointSourceType
    var basePath: String?
    var transformerStageNames: [String]
    var vaeDecoderModelName: String
    var embeddingsFileName: String

    static let defaultTransformerStageNames: [String] = (0..<7).map {
        "ZImageTurbo_TransformerBackbone_stage\($0).mlmodelc"
    }

    static let bundledDefault = ZImageCheckpointSet(
        id: "bundle.default",
        displayName: "Bundled ZImage",
        sourceType: .bundle,
        basePath: nil,
        transformerStageNames: defaultTransformerStageNames,
        vaeDecoderModelName: "VAEDecoder.mlmodelc",
        embeddingsFileName: "zimage_embeddings.bin"
    )

    var isBundled: Bool { sourceType == .bundle }
}

final class ZImageCheckpointStore: ObservableObject {
    @Published private(set) var checkpointSets: [ZImageCheckpointSet] = []
    @Published var selectedCheckpointID: String

    private let defaults = UserDefaults.standard
    private let setsKey = "zimage.checkpoint.sets"
    private let selectedKey = "zimage.checkpoint.selected"

    init() {
        self.selectedCheckpointID = defaults.string(forKey: selectedKey) ?? ZImageCheckpointSet.bundledDefault.id
        self.checkpointSets = Self.loadSets(from: defaults, key: setsKey)
        ensureBundledDefaultExists()
        normalizeSelection()
        persist()
    }

    var selectedCheckpoint: ZImageCheckpointSet? {
        checkpointSets.first(where: { $0.id == selectedCheckpointID })
    }

    func selectCheckpoint(id: String) {
        selectedCheckpointID = id
        normalizeSelection()
        persist()
    }

    @available(iOS 18.0, macOS 14.0, *)
    func loadSelectedAppPipeline(
        computeUnits: ComputeUnits,
        embeddingsOverrideURL: URL? = nil,
        runSmokeTest: Bool = true,
        smokeSteps: Int = 4,
        smokeSeed: UInt32 = 42
    ) throws -> AppPipeline {
        guard let bundleBaseURL = Bundle.main.resourceURL else {
            throw "Bundle resource URL is unavailable."
        }
        guard let selected = selectedCheckpoint else {
            throw "No checkpoint set selected."
        }

        let baseURL: URL
        if selected.isBundled {
            baseURL = bundleBaseURL
        } else {
            guard let basePath = selected.basePath else {
                throw "Imported checkpoint path is missing."
            }
            baseURL = URL(fileURLWithPath: basePath)
        }

        let effectiveEmbeddingsURL = embeddingsOverrideURL ?? baseURL.appending(path: selected.embeddingsFileName)
        let stageURLs = selected.transformerStageNames.map { baseURL.appending(path: $0) }
        let bootstrap = ZImageBootstrapConfig(
            transformerStageURLs: stageURLs,
            vaeDecoderURL: baseURL.appending(path: selected.vaeDecoderModelName),
            embeddingsURL: effectiveEmbeddingsURL
        )
        let loader = ZImagePipelineLoader(
            config: bootstrap,
            computeUnits: computeUnits
        )
        if runSmokeTest {
            _ = try loader.runSmokeTest(stepCount: smokeSteps, seed: smokeSeed)
        }
        let pipeline = try loader.load()
        return ZImageAppPipeline(
            pipeline: pipeline,
            transformerStageURLs: bootstrap.transformerStageURLs,
            vaeDecoderURL: bootstrap.vaeDecoderURL,
            embeddingsURL: bootstrap.embeddingsURL
        )
    }

    @discardableResult
    func importCheckpointFolder(from sourceURL: URL, displayName: String? = nil) throws -> ZImageCheckpointSet {
        let targetRoot = Settings.shared.applicationSupportURL().appendingPathComponent("zimage-checkpoints")
        try FileManager.default.createDirectory(at: targetRoot, withIntermediateDirectories: true)

        let baseName = displayName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? displayName!.trimmingCharacters(in: .whitespacesAndNewlines)
            : sourceURL.deletingPathExtension().lastPathComponent
        let sanitized = sanitizeFolderName(baseName)
        let destinationURL = uniqueDestination(root: targetRoot, preferredName: sanitized)
        try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        let discoveredStageNames = discoverTransformerStageNames(in: destinationURL)
        let discoveredVaeName = discoverFirstExisting(
            in: destinationURL,
            candidates: ["VAEDecoder.mlmodelc", "VAEDecoder.mlpackage"]
        )

        let newSet = ZImageCheckpointSet(
            id: UUID().uuidString,
            displayName: baseName.isEmpty ? "Imported \(checkpointSets.count + 1)" : baseName,
            sourceType: .imported,
            basePath: destinationURL.path,
            transformerStageNames: discoveredStageNames.isEmpty ? ZImageCheckpointSet.bundledDefault.transformerStageNames : discoveredStageNames,
            vaeDecoderModelName: discoveredVaeName ?? ZImageCheckpointSet.bundledDefault.vaeDecoderModelName,
            embeddingsFileName: ZImageCheckpointSet.bundledDefault.embeddingsFileName
        )
        checkpointSets.append(newSet)
        selectedCheckpointID = newSet.id
        sortSets()
        persist()
        return newSet
    }

    func removeImportedCheckpoint(id: String) {
        guard let set = checkpointSets.first(where: { $0.id == id && $0.sourceType == .imported }) else {
            return
        }
        if let basePath = set.basePath {
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: basePath))
        }
        checkpointSets.removeAll { $0.id == id }
        normalizeSelection()
        persist()
    }

    private func ensureBundledDefaultExists() {
        if checkpointSets.contains(where: { $0.id == ZImageCheckpointSet.bundledDefault.id }) {
            return
        }
        checkpointSets.insert(.bundledDefault, at: 0)
    }

    private func normalizeSelection() {
        if checkpointSets.contains(where: { $0.id == selectedCheckpointID }) {
            return
        }
        selectedCheckpointID = ZImageCheckpointSet.bundledDefault.id
    }

    private func sortSets() {
        checkpointSets.sort {
            if $0.sourceType == $1.sourceType {
                return $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
            }
            return $0.sourceType == .bundle
        }
    }

    private func persist() {
        defaults.set(selectedCheckpointID, forKey: selectedKey)
        if let data = try? JSONEncoder().encode(checkpointSets) {
            defaults.set(data, forKey: setsKey)
        }
    }

    private static func loadSets(from defaults: UserDefaults, key: String) -> [ZImageCheckpointSet] {
        guard let data = defaults.data(forKey: key),
              let decoded = try? JSONDecoder().decode([ZImageCheckpointSet].self, from: data) else {
            return [.bundledDefault]
        }
        return decoded
    }

    private func sanitizeFolderName(_ input: String) -> String {
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        let filtered = String(input.map { allowed.contains($0) ? $0 : "-" })
        let collapsed = filtered.replacingOccurrences(of: "--+", with: "-", options: .regularExpression)
        return collapsed.trimmingCharacters(in: CharacterSet(charactersIn: "-")).isEmpty ? "checkpoint" : collapsed
    }

    private func uniqueDestination(root: URL, preferredName: String) -> URL {
        var candidate = root.appendingPathComponent(preferredName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }
        var counter = 2
        while true {
            let url = root.appendingPathComponent("\(preferredName)-\(counter)", isDirectory: true)
            if !FileManager.default.fileExists(atPath: url.path) {
                candidate = url
                break
            }
            counter += 1
        }
        return candidate
    }

    private func discoverFirstExisting(in baseURL: URL, candidates: [String]) -> String? {
        let fm = FileManager.default
        for name in candidates {
            if fm.fileExists(atPath: baseURL.appending(path: name).path) {
                return name
            }
        }
        return nil
    }

    private func discoverTransformerStageNames(in baseURL: URL) -> [String] {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: baseURL.path) else {
            return []
        }

        let regexPattern = #"^ZImageTurbo_TransformerBackbone_stage(\d+)\.(mlmodelc|mlpackage)$"#
        guard let regex = try? NSRegularExpression(pattern: regexPattern) else {
            return []
        }

        var indexed: [(index: Int, name: String)] = []
        for name in names {
            let range = NSRange(location: 0, length: name.utf16.count)
            guard let match = regex.firstMatch(in: name, options: [], range: range),
                  match.numberOfRanges >= 2,
                  let idxRange = Range(match.range(at: 1), in: name),
                  let idx = Int(name[idxRange]) else {
                continue
            }
            indexed.append((index: idx, name: name))
        }
        if indexed.isEmpty {
            return []
        }

        indexed.sort { $0.index < $1.index }
        for (expected, found) in indexed.enumerated() where expected != found.index {
            return []
        }
        return indexed.map(\.name)
    }
}
