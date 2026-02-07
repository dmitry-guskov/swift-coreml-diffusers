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
        let fileDate = (try? fileURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
        let createdAt = metadata.date ?? fileDate ?? .distantPast
        let prompt = metadata.prompt.isEmpty ? "Generated image" : metadata.prompt.replacingOccurrences(of: "_", with: " ")

        return HistoryItem(
            id: name,
            fileURL: fileURL,
            prompt: prompt,
            seed: metadata.seed,
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
            reload()
        } catch {
            print("Error saving generated image history: \(error)")
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

struct HistoryGalleryView: View {
    @EnvironmentObject var historyStore: HistoryStore

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 12)]

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
                                HistoryImageCard(item: item)
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

    func submit() {
        if case .running = generation.state { return }
        Task {
            generation.state = .running(nil)
            do {
                let result = try await generation.generate()
                generation.state = .complete(generation.positivePrompt, result.image, result.lastSeed, result.interval)
                if let image = result.image {
                    historyStore.save(image: image, prompt: generation.positivePrompt, seed: result.lastSeed)
                }
            } catch {
                generation.state = .failed(error)
            }
        }
    }
    
    var body: some View {
        VStack {
            HStack {
                PromptTextField(text: $generation.positivePrompt, isPositivePrompt: true, model: iosModel().modelVersion)
                Button("Generate") {
                    submit()
                }
                .padding()
                .buttonStyle(.borderedProminent)
            }
            ImageWithPlaceholder(state: $generation.state)
                .scaledToFit()
            Spacer()
        }
        .padding()
        .environmentObject(generation)
    }
}

struct TextToImage: View {
    @EnvironmentObject var generation: GenerationContext
    @StateObject private var historyStore = HistoryStore()

    var body: some View {
        TabView {
            GenerationView()
                .tabItem {
                    Label("Generation", systemImage: "wand.and.stars")
                }
            HistoryGalleryView()
                .tabItem {
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
        }
        .environmentObject(generation)
        .environmentObject(historyStore)
    }
}
