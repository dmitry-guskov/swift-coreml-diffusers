//
//  DiscreteFlowScheduler.swift
//  stable-diffusion
//
//  Created by Dmitry Guskov on 10.02.2026.
//

import Accelerate
import CoreML

/// Euler discrete scheduler for flow-matching diffusion (Z-Image).
///
/// Translates `FlowMatchEulerDiscreteScheduler` from
/// `Z-Image/src/zimage/scheduler.py`.
///
/// ### Flow-matching ODE
/// The model learns a velocity field *v(x_t, t)* along the probability path
/// from noise (σ = 1) to data (σ = 0).  Sampling integrates the ODE
/// backwards:
///
/// ```
/// x_{t-1} = x_t + (σ_next − σ) · v(x_t, t)
/// ```
///
/// The schedule of σ values is optionally "time-shifted" to concentrate
/// more steps near higher noise levels, improving few-step quality.
///
/// ### Python defaults (from `config/model.py`)
/// ```python
/// DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS = 1000
/// DEFAULT_SCHEDULER_SHIFT = 3.0
/// DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING = False
/// ```
@available(iOS 16.2, macOS 13.1, *)
public final class DiscreteFlowScheduler: Scheduler {

    // MARK: - Scheduler protocol conformance

    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let timeSteps: [Int]
    public let initNoiseSigma: Float = 1.0

    /// Not used in flow-matching — stubs for `Scheduler` protocol
    public let betas:        [Float] = []
    public let alphas:       [Float] = []
    public let alphasCumProd: [Float] = []

    /// After each `step()`, the resulting denoised-so-far sample is appended.
    /// `modelOutputs.last` is the final latent after the last step.
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    // MARK: - Flow-matching state

    /// Full sigma schedule including the terminal 0
    /// Length = `inferenceStepCount + 1`
    let sigmas: [Float]

    /// Internal step counter — incremented by `step()`
    private var stepIndex: Int = 0

    // MARK: - Init

    /// Create a flow-matching Euler scheduler.
    ///
    /// Mirrors `FlowMatchEulerDiscreteScheduler.__init__` +
    /// `set_timesteps(num_inference_steps)` with
    /// `scheduler.sigma_min = 0.0` (as the Python pipeline sets).
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference (denoising) steps.
    ///   - trainStepCount: Number of training timesteps (default 1000).
    ///   - timeStepShift: Static time-shift factor (default 3.0 for Z-Image).
    public init(
        stepCount: Int = 4,
        trainStepCount: Int = 1000,
        timeStepShift: Float = 3.0
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount

        let shift = timeStepShift

        // ── Reproduce __init__: compute sigma_max from the full training schedule ──
        //
        //  Python:
        //    timesteps = np.linspace(1, 1000, 1000)[::-1]          # [1000 … 1]
        //    sigmas    = timesteps / 1000                           # [1.0 … 0.001]
        //    sigmas    = shift * sigmas / (1 + (shift-1) * sigmas)  # time-shifted
        //    sigma_max = sigmas[0]
        //
        // For shift = 3: sigma_max = 3·1/(1+2·1) = 1.0
        // (True for any shift because raw sigma_max = 1 → shifted = shift/shift = 1)
        let sigmaMax: Float = {
            let raw: Float = 1.0 // linspace(1, T, T)[0] / T = 1.0
            return shift * raw / (1.0 + (shift - 1.0) * raw)
        }()

        // Pipeline forces sigma_min = 0  (scheduler.sigma_min = 0.0)
        let sigmaMin: Float = 0.0

        // ── Reproduce set_timesteps ──
        //
        //  timesteps = linspace(sigma_max*T, sigma_min*T, steps+1)[:-1]
        //  sigmas    = timesteps / T
        //  sigmas    = shift * sigmas / (1 + (shift-1) * sigmas)
        //  sigmas    = cat([sigmas, [0]])
        //  timesteps = sigmas[:-1] * T
        //
        let tMax = sigmaMax * Float(trainStepCount)
        let tMin = sigmaMin * Float(trainStepCount)
        let rawTimesteps = Array(linspace(tMax, tMin, stepCount + 1).dropLast())

        let rawSigmas = rawTimesteps.map { $0 / Float(trainStepCount) }
        var shiftedSigmas = rawSigmas.map { s -> Float in
            shift * s / (1.0 + (shift - 1.0) * s)
        }
        // Terminal sigma (denoised)
        shiftedSigmas.append(0.0)
        self.sigmas = shiftedSigmas                 // length = stepCount + 1

        // Integer timesteps used by Dit (for timestep normalization) and
        // by the protocol's calculateTimesteps / for-loop
        self.timeSteps = shiftedSigmas.dropLast().map { sigma in
            Int((sigma * Float(trainStepCount)).rounded())
        }
    }

    // MARK: - Step

    /// Euler step: `x_{next} = x + (σ_next − σ) · v`
    ///
    /// Matches Python `FlowMatchEulerDiscreteScheduler.step()`.
    ///
    /// - Parameters:
    ///   - output: Model velocity prediction (already negated by `Dit.predictNoise`).
    ///   - t: Current timestep (unused for the computation — included for protocol).
    ///   - s: Current noisy sample `x_t`.
    /// - Returns: Denoised sample `x_{t−1}`.
    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Int,
        sample s: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {

        let sigma     = sigmas[stepIndex]
        let sigmaNext = sigmas[stepIndex + 1]
        let dt = sigmaNext - sigma           // negative (moving toward σ = 0)

        let scalarCount = s.scalarCount
        let prevSample = MLShapedArray<Float32>(unsafeUninitializedShape: s.shape) { scalars, _ in
            s.withUnsafeShapedBufferPointer { sBuf, _, _ in
                output.withUnsafeShapedBufferPointer { oBuf, _, _ in
                    for i in 0..<scalarCount {
                        scalars.initializeElement(at: i, to: sBuf[i] + dt * oBuf[i])
                    }
                }
            }
        }

        modelOutputs.append(prevSample)
        stepIndex += 1
        return prevSample
    }
}
