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
    let noiseURL: URL?
    let noiseShape: [Int]?
}

private struct HistoryMetadata: Codable {
    let prompt: String
    let seed: UInt32
    let createdAt: Date
    let noiseFilename: String?
    let noiseShape: [Int]?
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

    private func noiseURL(for imageURL: URL) -> URL {
        imageURL.deletingPathExtension().appendingPathExtension("noise")
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
        let savedNoiseURL: URL?
        if let noiseFilename = sidecarMetadata?.noiseFilename {
            savedNoiseURL = fileURL.deletingLastPathComponent().appendingPathComponent(noiseFilename)
        } else {
            let legacyNoiseURL = noiseURL(for: fileURL)
            savedNoiseURL = fileManager.fileExists(atPath: legacyNoiseURL.path) ? legacyNoiseURL : nil
        }

        return HistoryItem(
            id: name,
            fileURL: fileURL,
            prompt: prompt,
            seed: seed,
            createdAt: createdAt,
            noiseURL: savedNoiseURL,
            noiseShape: sidecarMetadata?.noiseShape
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

    func loadNoise(for item: HistoryItem) -> (data: Data, shape: [Int])? {
        guard let noiseURL = item.noiseURL,
              let noiseShape = item.noiseShape,
              let noiseData = try? Data(contentsOf: noiseURL)
        else {
            return nil
        }
        return (noiseData, noiseShape)
    }

    func save(
        image: CGImage,
        prompt: String,
        seed: UInt32,
        initialNoiseData: Data?,
        initialNoiseShape: [Int]?
    ) {
        let directoryURL = historyDirectoryURL()
        let timestamp = Self.filenameDateFormatter.string(from: Date())
        let filename = "\(timestamp)__\(seed)__\(prompt.first200Safe).png"
        let fileURL = directoryURL.appendingPathComponent(filename)

        guard let imageData = UIImage(cgImage: image).pngData() else {
            return
        }

        do {
            try imageData.write(to: fileURL, options: .atomic)

            var noiseFilename: String? = nil
            if let initialNoiseData, initialNoiseShape != nil {
                let noiseFileURL = noiseURL(for: fileURL)
                try initialNoiseData.write(to: noiseFileURL, options: .atomic)
                noiseFilename = noiseFileURL.lastPathComponent
            }

            let sidecarURL = metadataURL(for: fileURL)
            let metadata = HistoryMetadata(
                prompt: prompt,
                seed: seed,
                createdAt: Date(),
                noiseFilename: noiseFilename,
                noiseShape: initialNoiseShape
            )
            if let metadataData = try? JSONEncoder().encode(metadata) {
                try? metadataData.write(to: sidecarURL, options: .atomic)
            }
            reload()
        } catch {
            print("Error saving generated image history: \(error)")
        }
    }

    private func removeIfExists(_ url: URL) {
        guard fileManager.fileExists(atPath: url.path) else {
            return
        }
        do {
            try fileManager.removeItem(at: url)
        } catch {
            print("Error removing history file \(url.lastPathComponent): \(error)")
        }
    }

    func delete(_ item: HistoryItem) {
        let imageURL = item.fileURL
        let sidecarURL = metadataURL(for: imageURL)
        let fallbackNoiseURL = noiseURL(for: imageURL)

        removeIfExists(imageURL)
        removeIfExists(sidecarURL)
        if let noiseURL = item.noiseURL {
            removeIfExists(noiseURL)
        } else {
            removeIfExists(fallbackNoiseURL)
        }

        reload()
    }
}

final class PromptHistoryStore: ObservableObject {
    @Published private(set) var prompts: [String] = []

    private let defaults = UserDefaults.standard
    private let key = "recent_prompt_history_v1"
    private let maxCount = 20

    init() {
        prompts = defaults.stringArray(forKey: key) ?? []
    }

    func record(_ prompt: String) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        prompts.removeAll { $0.caseInsensitiveCompare(trimmed) == .orderedSame }
        prompts.insert(trimmed, at: 0)

        if prompts.count > maxCount {
            prompts = Array(prompts.prefix(maxCount))
        }

        defaults.set(prompts, forKey: key)
    }
}

private func startGeneration(prompt: String, generation: GenerationContext, historyStore: HistoryStore) {
    startGeneration(prompt: prompt, generation: generation, historyStore: historyStore, baseSeed: nil, baseImage: nil, forceSeed: nil)
}

private func loadCGImage(from fileURL: URL) -> CGImage? {
    UIImage(contentsOfFile: fileURL.path)?.cgImage
}

private func startGeneration(
    prompt: String,
    generation: GenerationContext,
    historyStore: HistoryStore,
    baseSeed: UInt32?,
    baseImage: CGImage?,
    forceSeed: UInt32?
) {
    if case .running = generation.state { return }

    let promptToUse = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !promptToUse.isEmpty else { return }

    generation.positivePrompt = promptToUse

    Task {
        await MainActor.run {
            generation.state = .running(nil)
        }
        do {
            let result = try await generation.generate(
                prompt: promptToUse,
                baseSeed: baseSeed,
                baseImage: baseImage,
                forceSeed: forceSeed
            )
            await MainActor.run {
                generation.state = .complete(promptToUse, result.image, result.lastSeed, result.interval)
                generation.updateVariationBase(
                    seed: result.lastSeed,
                    image: result.image,
                    noiseData: result.initialNoiseData,
                    noiseShape: result.initialNoiseShape
                )
            }
            if let image = result.image {
                historyStore.save(
                    image: image,
                    prompt: promptToUse,
                    seed: result.lastSeed,
                    initialNoiseData: result.initialNoiseData,
                    initialNoiseShape: result.initialNoiseShape
                )
            }
        } catch {
            await MainActor.run {
                generation.state = .failed(error)
            }
        }
    }
}

struct HistoryImageCard: View {
    let item: HistoryItem
    var onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack(alignment: .topTrailing) {
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

                Button {
                    onDelete()
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 14, weight: .semibold))
                        .padding(8)
                        .background(.ultraThinMaterial, in: Circle())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
                .padding(8)
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
    let items: [HistoryItem]
    @Binding var selectedIndex: Int?
    var isGenerating: Bool
    var onRegenerate: (HistoryItem) -> Void
    var onDelete: (HistoryItem) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showSavedMessage = false
    @State private var dragOffset: CGFloat = 0
    @State private var isTransitioning = false

    private var currentIndex: Int? {
        guard let selectedIndex, items.indices.contains(selectedIndex) else {
            return nil
        }
        return selectedIndex
    }

    private func item(at index: Int) -> HistoryItem? {
        guard items.indices.contains(index) else { return nil }
        return items[index]
    }

    private func image(for item: HistoryItem) -> UIImage? {
        UIImage(contentsOfFile: item.fileURL.path)
    }

    private func saveToPhotos(_ image: UIImage?) {
        guard let image else { return }
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        withAnimation(.easeInOut(duration: 0.2)) {
            showSavedMessage = true
        }
    }

    private var canShowNext: Bool {
        guard let currentIndex else { return false }
        return currentIndex + 1 < items.count
    }

    private var canShowPrevious: Bool {
        guard let currentIndex else { return false }
        return currentIndex > 0
    }

    private func clampedTranslation(_ translation: CGFloat) -> CGFloat {
        if translation < 0, !canShowNext {
            return translation * 0.2
        }
        if translation > 0, !canShowPrevious {
            return translation * 0.2
        }
        return translation
    }

    private func handleDragChanged(_ value: DragGesture.Value) {
        guard !isTransitioning else { return }
        dragOffset = clampedTranslation(value.translation.height)
    }

    private func transition(by delta: Int, pageHeight: CGFloat) {
        guard let currentIndex else { return }
        isTransitioning = true
        showSavedMessage = false
        let targetOffset = delta > 0 ? -pageHeight : pageHeight

        withAnimation(.interactiveSpring(response: 0.28, dampingFraction: 0.88, blendDuration: 0.16)) {
            dragOffset = targetOffset
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.22) {
            var transaction = Transaction()
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                selectedIndex = currentIndex + delta
                dragOffset = 0
            }
            isTransitioning = false
        }
    }

    private func handleDragEnded(_ value: DragGesture.Value, pageHeight: CGFloat) {
        guard !isTransitioning else { return }

        let threshold = min(max(pageHeight * 0.18, 80), 180)
        let translation = dragOffset
        let predicted = clampedTranslation(value.predictedEndTranslation.height)

        if (translation < -threshold || predicted < -threshold), canShowNext {
            transition(by: 1, pageHeight: pageHeight)
            return
        }

        if (translation > threshold || predicted > threshold), canShowPrevious {
            transition(by: -1, pageHeight: pageHeight)
            return
        }

        withAnimation(.spring(response: 0.34, dampingFraction: 0.86)) {
            dragOffset = 0
        }
    }

    private func deleteCurrentItem() {
        guard let currentIndex, let item = item(at: currentIndex) else { return }
        let oldCount = items.count
        onDelete(item)
        showSavedMessage = false
        dragOffset = 0
        isTransitioning = false

        if oldCount <= 1 {
            self.selectedIndex = nil
            dismiss()
            return
        }

        self.selectedIndex = min(currentIndex, oldCount - 2)
    }

    @ViewBuilder
    private func pageView(for item: HistoryItem) -> some View {
        VStack(spacing: 16) {
            HStack {
                Button {
                    selectedIndex = nil
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
                Button {
                    deleteCurrentItem()
                } label: {
                    Image(systemName: "trash.fill")
                        .font(.system(size: 17, weight: .semibold))
                        .padding(10)
                        .background(.white.opacity(0.16), in: Circle())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal)
            .padding(.top, 8)

            Group {
                if let image = image(for: item) {
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
                Text("Swipe up or down to browse history")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.7))

                HStack(spacing: 10) {
                    Button {
                        saveToPhotos(image(for: item))
                    } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)

                    Button {
                        onRegenerate(item)
                        selectedIndex = nil
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

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.black.ignoresSafeArea()

                if let currentIndex {
                    ZStack {
                        if let previousItem = item(at: currentIndex - 1) {
                            pageView(for: previousItem)
                                .offset(y: -proxy.size.height + dragOffset)
                        }

                        if let nextItem = item(at: currentIndex + 1) {
                            pageView(for: nextItem)
                                .offset(y: proxy.size.height + dragOffset)
                        }

                        if let currentItem = item(at: currentIndex) {
                            pageView(for: currentItem)
                                .offset(y: dragOffset)
                        }
                    }
                    .frame(width: proxy.size.width, height: proxy.size.height)
                    .clipped()
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 10)
                            .onChanged { value in
                                handleDragChanged(value)
                            }
                            .onEnded { value in
                                handleDragEnded(value, pageHeight: proxy.size.height)
                            }
                    )
                } else {
                    ProgressView()
                        .tint(.white)
                }
            }
        }
        .onChange(of: items.count) { newCount in
            guard let selectedIndex else { return }
            if newCount == 0 {
                self.selectedIndex = nil
                dismiss()
            } else if selectedIndex >= newCount {
                self.selectedIndex = newCount - 1
            }
        }
    }
}

struct HistoryGalleryView: View {
    @EnvironmentObject var historyStore: HistoryStore
    @EnvironmentObject var generation: GenerationContext
    @Binding var selectedTab: HomeTab
    @State private var selectedIndex: Int?

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
                                HistoryImageCard(item: item) {
                                    historyStore.delete(item)
                                }
                                .contentShape(RoundedRectangle(cornerRadius: 12))
                                .onTapGesture {
                                    guard let index = historyStore.items.firstIndex(where: { $0.id == item.id }) else {
                                        return
                                    }
                                    selectedIndex = index
                                }
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
            .fullScreenCover(
                isPresented: Binding(
                    get: { selectedIndex != nil },
                    set: { isPresented in
                        if !isPresented {
                            selectedIndex = nil
                        }
                    }
                )
            ) {
                HistoryImageDetailView(
                    items: historyStore.items,
                    selectedIndex: $selectedIndex,
                    isGenerating: isGenerating,
                    onRegenerate: { selectedItem in
                        selectedTab = .generation
                        generation.variationAmount = 0
                        let selectedImage = loadCGImage(from: selectedItem.fileURL)
                        let savedNoise = historyStore.loadNoise(for: selectedItem)
                        generation.loadHistorySelection(
                            prompt: selectedItem.prompt,
                            seed: selectedItem.seed,
                            image: selectedImage,
                            noiseData: savedNoise?.data,
                            noiseShape: savedNoise?.shape
                        )
                    },
                    onDelete: { item in
                        historyStore.delete(item)
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
    @EnvironmentObject var promptHistoryStore: PromptHistoryStore
    @FocusState private var promptFieldFocused: Bool
    @State private var showGenerationSettings = false

    private var isRunning: Bool {
        if case .running = generation.state {
            return true
        }
        return false
    }

    private var promptIsValid: Bool {
        !generation.positivePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var recentPrompts: [String] {
        Array(promptHistoryStore.prompts.prefix(8))
    }

    private var hasVariationSource: Bool {
        generation.variationBaseNoiseData != nil && generation.variationBaseNoiseShape != nil
    }

    private var variationValueText: String {
        String(format: "%.2f", generation.variationAmount)
    }

    private var stepCountValue: Int {
        max(1, min(50, Int(generation.steps.rounded())))
    }

    private var guidanceScaleValue: Double {
        max(0, min(20, generation.guidanceScale))
    }

    private func setStepCount(_ value: Int) {
        let clamped = max(1, min(50, value))
        let asDouble = Double(clamped)
        generation.steps = asDouble
        Settings.shared.stepCount = asDouble
    }

    private func setGuidanceScale(_ value: Double) {
        let clamped = max(0, min(20, value))
        generation.guidanceScale = clamped
        Settings.shared.guidanceScale = clamped
    }

    private func setNegativePrompt(_ value: String) {
        generation.negativePrompt = value
        Settings.shared.negativePrompt = value
    }

    private var variationDescription: String {
        let value = generation.variationAmount
        if value <= 0.0001 {
            return "0.00 reuses the loaded initial noise tensor."
        }
        if value >= 0.9999 {
            return "1.00 starts from independent noise."
        }
        return "Values between 0 and 1 interpolate loaded and new noise."
    }

    private func dismissKeyboard() {
        promptFieldFocused = false
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    private func usePrompt(_ prompt: String) {
        generation.positivePrompt = prompt
        Settings.shared.prompt = prompt
        dismissKeyboard()
    }

    private func submit(prompt overridePrompt: String? = nil) {
        let finalPrompt = (overridePrompt ?? generation.positivePrompt).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !finalPrompt.isEmpty else { return }

        generation.positivePrompt = finalPrompt
        Settings.shared.prompt = finalPrompt
        promptHistoryStore.record(finalPrompt)
        dismissKeyboard()

        startGeneration(prompt: finalPrompt, generation: generation, historyStore: historyStore)
    }

    private func repeatPrompt(_ prompt: String) {
        usePrompt(prompt)
        submit(prompt: prompt)
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                ImageWithPlaceholder(state: $generation.state)
                    .scaledToFit()
                    .frame(maxWidth: .infinity)

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Prompt")
                            .font(.headline)
                        Spacer()
                        Button {
                            dismissKeyboard()
                        } label: {
                            Label("Done", systemImage: "keyboard.chevron.compact.down")
                        }
                        .buttonStyle(.bordered)
                    }

                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $generation.positivePrompt)
                            .focused($promptFieldFocused)
                            .frame(minHeight: 120)
                            .padding(8)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .strokeBorder(Color.secondary.opacity(0.35), lineWidth: 1)
                            )

                        if generation.positivePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text("Describe what you want to generate...")
                                .foregroundColor(.secondary)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 18)
                                .allowsHitTesting(false)
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Variation")
                                .font(.subheadline.weight(.semibold))
                            Spacer()
                            Text(variationValueText)
                                .font(.subheadline.monospacedDigit())
                                .foregroundColor(.secondary)
                        }

                        Slider(value: $generation.variationAmount, in: 0...1)

                        Text(variationDescription)
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if generation.variationAmount < 1 && !hasVariationSource {
                            Text("Generate an image first or pick one from History to use as a source.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    HStack {
                        Button("Clear") {
                            generation.positivePrompt = ""
                            Settings.shared.prompt = ""
                        }
                        .buttonStyle(.bordered)

                        Spacer()

                        Button {
                            submit()
                        } label: {
                            Label(isRunning ? "Generating..." : "Generate", systemImage: "sparkles")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isRunning || !promptIsValid)
                    }
                }

                if !recentPrompts.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recent Prompts")
                            .font(.headline)

                        ForEach(recentPrompts, id: \.self) { prompt in
                            VStack(alignment: .leading, spacing: 8) {
                                Text(prompt)
                                    .font(.subheadline)
                                    .lineLimit(3)

                                HStack {
                                    Button("Use") {
                                        usePrompt(prompt)
                                    }
                                    .buttonStyle(.bordered)

                                    Button {
                                        repeatPrompt(prompt)
                                    } label: {
                                        Label("Repeat", systemImage: "arrow.clockwise")
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(isRunning)
                                }
                            }
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                        }
                    }
                }
            }
            .padding()
        }
        .onChange(of: generation.positivePrompt) { newPrompt in
            Settings.shared.prompt = newPrompt
        }
        .scrollDismissesKeyboard(.interactively)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    dismissKeyboard()
                }
            }
        }
        .overlay(alignment: .topTrailing) {
            Button {
                dismissKeyboard()
                showGenerationSettings = true
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.headline)
                    .padding(10)
                    .background(.thinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Generation settings")
            .padding(.top, 8)
            .padding(.trailing, 16)
        }
        .sheet(isPresented: $showGenerationSettings) {
            GenerationSettingsSheet(
                steps: stepCountValue,
                guidanceScale: guidanceScaleValue,
                negativePrompt: generation.negativePrompt,
                onChangeSteps: setStepCount(_:),
                onChangeGuidanceScale: setGuidanceScale(_:),
                onChangeNegativePrompt: setNegativePrompt(_:)
            )
        }
        .environmentObject(generation)
    }
}

private struct GenerationSettingsSheet: View {
    @Environment(\.dismiss) private var dismiss
    let steps: Int
    let guidanceScale: Double
    let negativePrompt: String
    var onChangeSteps: (Int) -> Void
    var onChangeGuidanceScale: (Double) -> Void
    var onChangeNegativePrompt: (String) -> Void

    var body: some View {
        NavigationView {
            Form {
                Section("Steps") {
                    HStack {
                        Text("Number of steps")
                        Spacer()
                        Text("\(steps)")
                            .font(.body.monospacedDigit())
                            .foregroundColor(.secondary)
                    }

                    Stepper(value: Binding(
                        get: { steps },
                        set: { newValue in
                            onChangeSteps(newValue)
                        }
                    ), in: 1...50) {
                        Text("Adjust steps")
                    }

                    Slider(
                        value: Binding(
                            get: { Double(steps) },
                            set: { newValue in
                                onChangeSteps(Int(newValue.rounded()))
                            }
                        ),
                        in: 1...50,
                        step: 1
                    )
                }

                Section("Guidance") {
                    HStack {
                        Text("CFG scale")
                        Spacer()
                        Text(String(format: "%.1f", guidanceScale))
                            .font(.body.monospacedDigit())
                            .foregroundColor(.secondary)
                    }

                    Stepper(value: Binding(
                        get: { guidanceScale },
                        set: { newValue in
                            onChangeGuidanceScale(newValue)
                        }
                    ), in: 0...20, step: 0.1) {
                        Text("Adjust CFG scale")
                    }

                    Slider(
                        value: Binding(
                            get: { guidanceScale },
                            set: { newValue in
                                onChangeGuidanceScale(newValue)
                            }
                        ),
                        in: 0...20,
                        step: 0.1
                    )
                }

                Section("Negative Prompt") {
                    TextEditor(text: Binding(
                        get: { negativePrompt },
                        set: { newValue in
                            onChangeNegativePrompt(newValue)
                        }
                    ))
                    .frame(minHeight: 110)
                }
            }
            .navigationTitle("Generation Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct TextToImage: View {
    @EnvironmentObject var generation: GenerationContext
    @StateObject private var historyStore = HistoryStore()
    @StateObject private var promptHistoryStore = PromptHistoryStore()
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
        .environmentObject(promptHistoryStore)
    }
}
