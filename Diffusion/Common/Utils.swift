//
//  Utils.swift
//  Diffusion
//
//  Created by Pedro Cuenca on 14/1/23.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import Foundation

/// Copies a file or directory picked by the system document picker into a destination folder
/// inside the app container, where it can be accessed without any security scope.
///
/// On iOS, security-scoped bookmarks do NOT persist security grants across app launches —
/// only macOS supports `withSecurityScope` bookmarks. The only reliable strategy is to
/// import (copy) the item immediately while the picker's scope is still active.
///
/// NSFileCoordinator is used so that iCloud placeholder files are downloaded
/// transparently before the copy begins.
///
/// - Parameters:
///   - pickerURL: Security-scoped URL returned by `fileImporter` / `UIDocumentPickerViewController`.
///                The caller must have already called `startAccessingSecurityScopedResource()`.
///   - folder:    Destination directory inside the app container (will be created if absent).
///   - filename:  Optional override for the destination file/folder name.
///                Defaults to `pickerURL.lastPathComponent`.
/// - Returns: URL of the copy inside `folder`.
func importExternalResource(
    pickerURL: URL,
    into folder: URL,
    filename: String? = nil
) throws -> URL {
    let fm = FileManager.default
    try fm.createDirectory(at: folder, withIntermediateDirectories: true)

    let destName = filename ?? pickerURL.lastPathComponent
    let destURL = folder.appendingPathComponent(destName)

    // Remove stale copy so we always get a fresh import.
    try? fm.removeItem(at: destURL)

    var coordError: NSError?
    var copyError: Error?
    // NSFileCoordinator handles iCloud files: it blocks until the item is
    // fully downloaded locally before invoking the accessor block.
    NSFileCoordinator().coordinate(
        readingItemAt: pickerURL,
        options: .withoutChanges,
        error: &coordError
    ) { srcURL in
        do {
            try fm.copyItem(at: srcURL, to: destURL)
        } catch {
            copyError = error
        }
    }
    if let err = (coordError ?? copyError) { throw err }
    return destURL
}

extension String: @retroactive Error {}

extension Bundle {
    var displayName: String? {
        object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
            ?? object(forInfoDictionaryKey: "CFBundleName") as? String
    }
}

extension Double {
    func formatted(_ format: String) -> String {
        return String(format: "\(format)", self)
    }
}

extension String {
    var first200Safe: String {
        let endIndex = index(startIndex, offsetBy: Swift.min(200, count))
        let substring = String(self[startIndex..<endIndex])
        
        // Replace whitespace with underscore or dash
        let replacedSubstring = substring
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "\t", with: "_")
            .replacingOccurrences(of: "\n", with: "_")
        
        // Remove unsafe characters from the substring
        let allowedCharacters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let filteredSubstring = replacedSubstring
            .components(separatedBy: allowedCharacters.inverted)
            .joined()
        
        return filteredSubstring
    }
}

/// Returns an array of booleans that indicates at which steps a preview should be generated.
///
/// - Parameters:
///   - numInferenceSteps: The total number of inference steps.
///   - numPreviews: The desired number of previews.
///
/// - Returns: An array of booleans of size `numInferenceSteps`, where `true` values represent steps at which a preview should be made.
func previewIndices(_ numInferenceSteps: Int, _ numPreviews: Int) -> [Bool] {
    // Ensure valid parameters
    guard numInferenceSteps > 0, numPreviews > 0 else {
        return [Bool](repeating: false, count: numInferenceSteps)
    }

    // Compute the ideal (floating-point) step size, which represents the average number of steps between previews
    let idealStep = Double(numInferenceSteps) / Double(numPreviews)

    // Compute the actual steps at which previews should be made. For each preview, we multiply the ideal step size by the preview number, and round to the nearest integer.
    // The result is converted to a `Set` for fast membership tests.
    let previewIndices: Set<Int> = Set((0..<numPreviews).map { previewIndex in
        return Int(round(Double(previewIndex) * idealStep))
    })
    
    // Construct an array of booleans where each value indicates whether or not a preview should be made at that step.
    let previewArray = (0..<numInferenceSteps).map { previewIndices.contains($0) }

    return previewArray
}

extension Double {
    func reduceScale(to places: Int) -> Double {
        let multiplier = pow(10, Double(places))
        let newDecimal = multiplier * self // move the decimal right
        let truncated = Double(Int(newDecimal)) // drop the fraction
        let originalDecimal = truncated / multiplier // move the decimal back
        return originalDecimal
    }
}

func formatLargeNumber(_ n: UInt32) -> String {
    let num = abs(Double(n))

    switch num {
    case 1_000_000_000...:
        var formatted = num / 1_000_000_000
        formatted = formatted.reduceScale(to: 3)
        return "\(formatted)B"

    case 1_000_000...:
        var formatted = num / 1_000_000
        formatted = formatted.reduceScale(to: 3)
        return "\(formatted)M"

    case 1_000...:
        return "\(n)"

    case 0...:
        return "\(n)"

    default:
        return "\(n)"
    }
}
