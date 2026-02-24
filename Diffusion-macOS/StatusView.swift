//
//  StatusView.swift
//  Diffusion-macOS
//
//  Created by Cyril Zakka on 1/12/23.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import SwiftUI

struct StatusView: View {
    @EnvironmentObject var generation: GenerationContext
    var pipelineState: Binding<PipelineState>
    
    @State private var showErrorPopover = false
    @State private var loadingStartDate: Date?

    private var pipelineReady: Bool {
        if case .ready = pipelineState.wrappedValue {
            return true
        }
        return false
    }

    private var generationRunning: Bool {
        if case .running = generation.state {
            return true
        }
        return false
    }

    private var pipelineLoading: Bool {
        if case .loading = pipelineState.wrappedValue {
            return true
        }
        return false
    }
    
    func submit() {
        if case .running = generation.state { return }
        Task {
            generation.state = .running(nil)
            do {
                let result = try await generation.generate()
                if result.userCanceled {
                    generation.state = .userCanceled
                } else {
                    generation.state = .complete(generation.positivePrompt, result.image, result.lastSeed, result.interval)
                }
            } catch {
                generation.state = .failed(error)
            }
        }
    }

    func errorWithDetails(_ message: String, error: Error) -> any View {
        HStack {
            Text(message)
            Spacer()
            Button {
                showErrorPopover.toggle()
            } label: {
                Image(systemName: "info.circle")
            }.buttonStyle(.plain)
            .popover(isPresented: $showErrorPopover) {
                VStack {
                    Text(verbatim: "\(error)")
                    .lineLimit(nil)
                    .padding(.all, 5)
                    Button {
                        showErrorPopover.toggle()
                    } label: {
                        Text("Dismiss").frame(maxWidth: 200)
                    }
                    .padding(.bottom)
                }
                .frame(minWidth: 400, idealWidth: 400, maxWidth: 400)
                .fixedSize()
            }
        }
    }

    func generationStatusView() -> any View {
        switch generation.state {
        case .startup: return EmptyView()
        case .running(let progress):
            guard let progress = progress, progress.stepCount > 0 else {
                // The first time it takes a little bit before generation starts
                return HStack {
                    Text(generation.generationProgressSnapshot.phaseText)
                    Spacer()
                }
            }
            let snapshot = generation.generationProgressSnapshot
            let itPerSecText = snapshot.iterationsPerSecond.map { String(format: "%.2f it/s", $0) } ?? "it/s pending"
            let etaText = snapshot.etaSeconds.map { formatDuration($0) } ?? "pending"
            let elapsedText = formatDuration(snapshot.elapsedSeconds)
            let percentText = String(format: "%.1f%%", snapshot.fraction * 100)
            return HStack {
                Text("Step \(snapshot.step)/\(snapshot.stepCount) | \(percentText) | \(itPerSecText) | ETA \(etaText) | Elapsed \(elapsedText)")
                Spacer()
            }
        case .complete(_, let image, let lastSeed, let interval):
            guard let _ = image else {
                return HStack {
                    Text("Safety checker triggered, please try a different prompt or seed.")
                    Spacer()
                }
            }
                              
            return HStack {
                let intervalString = String(format: "Time: %.1fs", interval ?? 0)
                Text(intervalString)
                Spacer()
                if generation.seed != lastSeed {
                    
                    Text(String("Seed: \(formatLargeNumber(lastSeed))"))
                    Button("Set") {
                        generation.seed = lastSeed
                    }
                }
            }.frame(maxHeight: 25)
        case .failed(let error):
            return errorWithDetails("Generation error", error: error)
        case .userCanceled:
            return HStack {
                Text("Generation canceled.")
                Spacer()
            }
        }
    }
    
    @ViewBuilder
    private func pipelineStatusView() -> some View {
        switch pipelineState.wrappedValue {
        case .downloading(let progress):
            ProgressView("Downloading…", value: progress*100, total: 110)
                .padding(.bottom, 4)
            Text("Generate is disabled until model resources finish downloading.")
                .font(.caption)
                .foregroundColor(.secondary)
        case .uncompressing:
            ProgressView("Uncompressing…", value: 100, total: 110)
                .padding(.bottom, 4)
            Text("Generate is disabled until model resources are uncompressed.")
                .font(.caption)
                .foregroundColor(.secondary)
        case .loading:
            ProgressView("Loading…", value: 105, total: 110)
                .padding(.bottom, 4)
            let elapsed = loadingStartDate.map { formatDuration(Date().timeIntervalSince($0)) } ?? "00:00"
            Text("Preparing models | Transformer+VAE+embeddings | ETA pending | Elapsed \(elapsed)")
                .font(.caption)
                .foregroundColor(.secondary)
        case .ready:
            AnyView(generationStatusView())
        case .failed(let error):
            AnyView(errorWithDetails("Pipeline loading error", error: error))
            Text("Select valid Transformer/VAE paths and click Reload Models.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                submit()
            } label: {
                Text(generationRunning ? "Generating..." : "Generate")
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!pipelineReady || generationRunning)

            pipelineStatusView()
        }
        .onAppear {
            loadingStartDate = pipelineLoading ? Date() : nil
        }
        .onChange(of: pipelineLoading) { _, isLoading in
            loadingStartDate = isLoading ? Date() : nil
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        let minutes = total / 60
        let remainder = total % 60
        return String(format: "%02d:%02d", minutes, remainder)
    }
}

struct StatusView_Previews: PreviewProvider {
    static var previews: some View {
        StatusView(pipelineState: .constant(.downloading(0.2)))
    }
}
