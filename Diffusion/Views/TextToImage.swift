//
//  TextToImage.swift
//  Diffusion
//
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import SwiftUI
import Combine
import StableDiffusion

struct HistoryItem: Identifiable {
    let id: String
    let fileURL: URL
    let prompt: String
    let seed: UInt32
    let createdAt: Date
}

private struct HistoryMetadata: Codable {
    let prompt: String
    let seed: UInt32
    let createdAt: Date
}

enum HomeTab: Hashable {
    case generation
    case history
}

final class HistoryStore: ObservableObject {
    @Published private(set) var items: [HistoryItem] = []

    private let fileManager = FileManager.default

    private static let filenameDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmssSSS"
        return formatter
    }()

    init() {
        reload()
    }

    private func historyDirectoryURL() -> URL {
        let directoryURL = Settings.shared.applicationSupportURL().appendingPathComponent("hf-diffusion-history")
        if !fileManager.fileExists(atPath: directoryURL.path) {
            do {
                try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            } catch {
                print("Error creating history directory: \(error)")
            }
        }
        return directoryURL
    }

    private func metadataURL(for imageURL: URL) -> URL {
        imageURL.deletingPathExtension().appendingPathExtension("json")
    }

    private func loadMetadata(for imageURL: URL) -> HistoryMetadata? {
        let sidecarURL = metadataURL(for: imageURL)
        guard let data = try? Data(contentsOf: sidecarURL) else {
            return nil
        }
        return try? JSONDecoder().decode(HistoryMetadata.self, from: data)
    }

    private func parseMetadata(from filename: String) -> (date: Date?, seed: UInt32, prompt: String) {
        guard let firstSeparator = filename.range(of: "__"),
              let secondSeparator = filename.range(of: "__", range: firstSeparator.upperBound..<filename.endIndex)
        else {
            return (nil, 0, filename)
        }

        let dateToken = String(filename[..<firstSeparator.lowerBound])
        let seedToken = String(filename[firstSeparator.upperBound..<secondSeparator.lowerBound])
        let promptToken = String(filename[secondSeparator.upperBound...])
        return (Self.filenameDateFormatter.date(from: dateToken), UInt32(seedToken) ?? 0, promptToken)
    }

    private func item(from fileURL: URL) -> HistoryItem {
        let name = fileURL.deletingPathExtension().lastPathComponent
        let metadata = parseMetadata(from: name)
        let sidecarMetadata = loadMetadata(for: fileURL)
        let fileDate = (try? fileURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
        let createdAt = sidecarMetadata?.createdAt ?? metadata.date ?? fileDate ?? .distantPast
        let prompt: String
        if let exactPrompt = sidecarMetadata?.prompt, !exactPrompt.isEmpty {
            prompt = exactPrompt
        } else {
            let parsedPrompt = metadata.prompt.replacingOccurrences(of: "_", with: " ")
            prompt = parsedPrompt.isEmpty ? "Generated image" : parsedPrompt
        }
        let seed = sidecarMetadata?.seed ?? metadata.seed

        return HistoryItem(
            id: name,
            fileURL: fileURL,
            prompt: prompt,
            seed: seed,
            createdAt: createdAt
        )
    }

    private func updateItemsOnMain(_ newItems: [HistoryItem]) {
        if Thread.isMainThread {
            self.items = newItems
        } else {
            DispatchQueue.main.async {
                self.items = newItems
            }
        }
    }

    func reload() {
        let directoryURL = historyDirectoryURL()
        let imageURLs = (try? fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let loadedItems = imageURLs
            .filter { $0.pathExtension.lowercased() == "png" }
            .map(item(from:))
            .sorted(by: { $0.createdAt > $1.createdAt })

        updateItemsOnMain(loadedItems)
    }

    func save(image: CGImage, prompt: String, seed: UInt32) {
        let directoryURL = historyDirectoryURL()
        let timestamp = Self.filenameDateFormatter.string(from: Date())
        let filename = "\(timestamp)__\(seed)__\(prompt.first200Safe).png"
        let fileURL = directoryURL.appendingPathComponent(filename)

        guard let imageData = UIImage(cgImage: image).pngData() else {
            return
        }

        do {
            try imageData.write(to: fileURL, options: .atomic)
            let sidecarURL = metadataURL(for: fileURL)
            let metadata = HistoryMetadata(prompt: prompt, seed: seed, createdAt: Date())
            if let metadataData = try? JSONEncoder().encode(metadata) {
                try? metadataData.write(to: sidecarURL, options: .atomic)
            }
            reload()
        } catch {
            print("Error saving generated image history: \(error)")
        }
    }
}

private func startGeneration(prompt: String, generation: GenerationContext, historyStore: HistoryStore) {
    if case .running = generation.state { return }

    let promptToUse = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !promptToUse.isEmpty else { return }

    generation.positivePrompt = promptToUse

    Task {
        generation.state = .running(nil)
        do {
            let result = try await generation.generate()
            generation.state = .complete(promptToUse, result.image, result.lastSeed, result.interval)
            if let image = result.image {
                historyStore.save(image: image, prompt: promptToUse, seed: result.lastSeed)
            }
        } catch {
            generation.state = .failed(error)
        }
    }
}

struct HistoryImageCard: View {
    let item: HistoryItem

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let image = UIImage(contentsOfFile: item.fileURL.path) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(height: 140)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                RoundedRectangle(cornerRadius: 10)
                    .fill(.gray.opacity(0.15))
                    .frame(height: 140)
                    .overlay(
                        Image(systemName: "photo")
                            .foregroundColor(.secondary)
                    )
            }

            Text(item.prompt)
                .font(.caption)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(item.createdAt.formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundColor(.secondary)
            Text("Seed \(item.seed)")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct HistoryImageDetailView: View {
    let item: HistoryItem
    var isGenerating: Bool
    var onRegenerate: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showSavedMessage = false

    private var image: UIImage? {
        UIImage(contentsOfFile: item.fileURL.path)
    }

    private func saveToPhotos() {
        guard let image = image else { return }
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        withAnimation(.easeInOut(duration: 0.2)) {
            showSavedMessage = true
        }
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 16) {
                HStack {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 28))
                            .foregroundStyle(.white.opacity(0.9))
                    }
                    Spacer()
                    if showSavedMessage {
                        Text("Saved")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(.white.opacity(0.18), in: Capsule())
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)

                Group {
                    if let image = image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                    } else {
                        RoundedRectangle(cornerRadius: 16)
                            .fill(.white.opacity(0.12))
                            .overlay(
                                Image(systemName: "photo")
                                    .font(.system(size: 40))
                                    .foregroundStyle(.white.opacity(0.75))
                            )
                    }
                }
                .padding(.horizontal)

                VStack(alignment: .leading, spacing: 10) {
                    Text(item.prompt)
                        .font(.body)
                        .foregroundStyle(.white)
                    Text(item.createdAt.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.8))
                    Text("Seed \(item.seed)")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.8))

                    HStack(spacing: 10) {
                        Button {
                            saveToPhotos()
                        } label: {
                            Label("Save", systemImage: "square.and.arrow.down")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.blue)

                        Button {
                            onRegenerate(item.prompt)
                            dismiss()
                        } label: {
                            Label("Regenerate", systemImage: "arrow.triangle.2.circlepath")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isGenerating)
                    }
                }
                .padding()
                .background(.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 16))
                .padding(.horizontal)
                .padding(.bottom, 20)
            }
        }
    }
}

struct HistoryGalleryView: View {
    @EnvironmentObject var historyStore: HistoryStore
    @EnvironmentObject var generation: GenerationContext
    @Binding var selectedTab: HomeTab
    @State private var selectedItem: HistoryItem?

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 12)]

    private var isGenerating: Bool {
        if case .running = generation.state {
            return true
        }
        return false
    }

    var body: some View {
        NavigationView {
            Group {
                if historyStore.items.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.stack")
                            .font(.system(size: 36))
                            .foregroundColor(.secondary)
                        Text("No generated images yet")
                            .font(.headline)
                        Text("Generate an image and it will appear here.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 12) {
                            ForEach(historyStore.items) { item in
                                Button {
                                    selectedItem = item
                                } label: {
                                    HistoryImageCard(item: item)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("History")
            .onAppear {
                historyStore.reload()
            }
            .fullScreenCover(item: $selectedItem) { item in
                HistoryImageDetailView(
                    item: item,
                    isGenerating: isGenerating,
                    onRegenerate: { prompt in
                        selectedTab = .generation
                        startGeneration(prompt: prompt, generation: generation, historyStore: historyStore)
                    }
                )
            }
        }
    }
}

/// Presents "Share" + "Save" buttons on Mac; just "Share" on iOS/iPadOS.
/// This is because I didn't find a way for "Share" to show a Save option when running on macOS.
struct ShareButtons: View {
    var image: CGImage
    var name: String
    
    var filename: String {
        name.replacingOccurrences(of: " ", with: "_")
    }
    
    var body: some View {
        let imageView = Image(image, scale: 1, label: Text(name))

        if runningOnMac {
            HStack {
                ShareLink(item: imageView, preview: SharePreview(name, image: imageView))
                Button() {
                    guard let imageData = UIImage(cgImage: image).pngData() else {
                        return
                    }
                    do {
                        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("\(filename).png")
                        try imageData.write(to: fileURL)
                        let controller = UIDocumentPickerViewController(forExporting: [fileURL])
                        
                        let scene = UIApplication.shared.connectedScenes.first as! UIWindowScene
                        scene.windows.first!.rootViewController!.present(controller, animated: true)
                    } catch {
                        print("Error creating file")
                    }
                } label: {
                    Label("Save…", systemImage: "square.and.arrow.down")
                }
            }
        } else {
            ShareLink(item: imageView, preview: SharePreview(name, image: imageView))
        }
    }
}

struct ImageWithPlaceholder: View {
    @EnvironmentObject var generation: GenerationContext
    var state: Binding<GenerationState>
        
    var body: some View {
        switch state.wrappedValue {
        case .startup: return AnyView(Image("placeholder").resizable())
        case .running(let progress):
            guard let progress = progress, progress.stepCount > 0 else {
                // The first time it takes a little bit before generation starts
                return AnyView(ProgressView())
            }

            let step = Int(progress.step) + 1
            let fraction = Double(step) / Double(progress.stepCount)
            let label = "Step \(step) of \(progress.stepCount)"
            return AnyView(VStack {
                Group {
                    if let safeImage = generation.previewImage {
                        Image(safeImage, scale: 1, label: Text("generated"))
                            .resizable()
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                    }
                }
                ProgressView(label, value: fraction, total: 1).padding()
            })
        case .complete(let lastPrompt, let image, _, let interval):
            guard let theImage = image else {
                return AnyView(Image(systemName: "exclamationmark.triangle").resizable())
            }
                              
            let imageView = Image(theImage, scale: 1, label: Text("generated"))
            return AnyView(
                VStack {
                    imageView.resizable().clipShape(RoundedRectangle(cornerRadius: 20))
                    HStack {
                        let intervalString = String(format: "Time: %.1fs", interval ?? 0)
                        Rectangle().fill(.clear).overlay(Text(intervalString).frame(maxWidth: .infinity, alignment: .leading).padding(.leading))
                        Rectangle().fill(.clear).overlay(
                            HStack {
                                Spacer()
                                ShareButtons(image: theImage, name: lastPrompt).padding(.trailing)
                            }
                        )
                    }.frame(maxHeight: 25)
            })
        case .failed(_):
            return AnyView(Image(systemName: "exclamationmark.triangle").resizable())
        case .userCanceled:
            return AnyView(Text("Generation canceled"))
        }
    }
}

struct GenerationView: View {
    @EnvironmentObject var generation: GenerationContext
    @EnvironmentObject var historyStore: HistoryStore

    private func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    func submit() {
        startGeneration(prompt: generation.positivePrompt, generation: generation, historyStore: historyStore)
    }
    
    var body: some View {
        VStack {
            ImageWithPlaceholder(state: $generation.state)
                .scaledToFit()

            HStack {
                PromptTextField(text: $generation.positivePrompt, isPositivePrompt: true, model: iosModel().modelVersion)
                Button("Generate") {
                    dismissKeyboard()
                    submit()
                }
                .padding()
                .buttonStyle(.borderedProminent)
                Button {
                    dismissKeyboard()
                } label: {
                    Image(systemName: "keyboard.chevron.compact.down")
                }
                .padding(.trailing, 4)
                .buttonStyle(.bordered)
            }
            Spacer()
        }
        .padding()
        .environmentObject(generation)
    }
}

struct TextToImage: View {
    @EnvironmentObject var generation: GenerationContext
    @StateObject private var historyStore = HistoryStore()
    @State private var selectedTab: HomeTab = .generation

    var body: some View {
        TabView(selection: $selectedTab) {
            GenerationView()
                .tabItem {
                    Label("Generation", systemImage: "wand.and.stars")
                }
                .tag(HomeTab.generation)
            HistoryGalleryView(selectedTab: $selectedTab)
                .tabItem {
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
                .tag(HomeTab.history)
        }
        .environmentObject(generation)
        .environmentObject(historyStore)
    }
}
