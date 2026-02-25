// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import CoreML
import Foundation

/// A class to manage and gate access to a Core ML model
///
/// It will automatically load a model into memory when needed or requested
/// It allows one to request to unload the model from memory
@available(iOS 16.2, macOS 13.1, *)
public final class ManagedMLModel: ResourceManaging {

    /// The location of the model
    var modelURL: URL

    /// The configuration to be used when the model is loaded
    var configuration: MLModelConfiguration

    /// The loaded model (when loaded)
    var loadedModel: MLModel?

    /// Queue to protect access to loaded model
    var queue: DispatchQueue
    
    /// Model name for logging
    private var modelName: String {
        modelURL.lastPathComponent
    }

    /// Create a managed model given its location and desired loaded configuration
    ///
    /// - Parameters:
    ///     - url: The location of the model
    ///     - configuration: The configuration to be used when the model is loaded/used
    /// - Returns: A managed model that has not been loaded
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.modelURL = url
        self.configuration = configuration
        self.loadedModel = nil
        self.queue = DispatchQueue(label: "managed.\(url.lastPathComponent)")
    }

    /// Instantiation and load model into memory
    public func loadResources() throws {
        try queue.sync {
            try loadModel()
        }
    }

    /// Unload the model if it was loaded
    public func unloadResources() {
        queue.sync {
            if loadedModel != nil {
                logMemory("ManagedMLModel[\(modelName)].unload.before")
                loadedModel = nil
                logMemory("ManagedMLModel[\(modelName)].unload.after")
            }
        }
    }

    /// Perform an operation with the managed model via a supplied closure.
    ///  The model will be loaded and supplied to the closure and should only be
    ///  used within the closure to ensure all resource management is synchronized
    ///
    /// - Parameters:
    ///     - body: Closure which performs and action on a loaded model
    /// - Returns: The result of the closure
    /// - Throws: An error if the model cannot be loaded or if the closure throws
    public func perform<R>(_ body: (MLModel) throws -> R) throws -> R {
        return try queue.sync {
            try autoreleasepool {
                try loadModel()
                return try body(loadedModel!)
            }
        }
    }

    private func loadModel() throws {
        if loadedModel == nil {
            logMemory("ManagedMLModel[\(modelName)].load.before")
            print("[ManagedMLModel] Loading model: \(modelName)")
            let startTime = CFAbsoluteTimeGetCurrent()
            loadedModel = try MLModel(contentsOf: modelURL,
                                      configuration: configuration)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            print("[ManagedMLModel] Loaded \(modelName) in \(String(format: "%.2f", elapsed))s")
            logMemory("ManagedMLModel[\(modelName)].load.after")
        }
    }
    
    private func logMemory(_ checkpoint: String) {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let usedMB = Double(info.resident_size) / 1_048_576
            print("[Memory] \(checkpoint): \(String(format: "%.1f", usedMB)) MB")
        }
    }
}
