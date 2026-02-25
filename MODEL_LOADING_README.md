# Z-Image Model Loading Architecture

This document describes how CoreML models are loaded and managed in the Z-Image generation pipeline.

## Overview

The Z-Image pipeline uses two CoreML models:

1. **DiT (Diffusion Transformer)** - `ZImageTurbo_TransformerBackbone_fp16.mlmodelc`
   - Size: ~300-500MB depending on quantization
   - Purpose: Predicts noise at each denoising step
   - Runs multiple times per generation (once per step)

2. **VAE Decoder** - `VAEDecoder.mlmodelc`
   - Size: ~80-150MB
   - Purpose: Decodes final latent to RGB image
   - Runs once per generation (at the end)

## Loading Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        App Startup                                   │
├─────────────────────────────────────────────────────────────────────┤
│  LoadingView.onAppear                                                │
│       │                                                              │
│       ▼                                                              │
│  ZImagePipelineLoader.loadAppPipeline()                             │
│       │                                                              │
│       ├──► validateResources() - Check files exist                  │
│       │                                                              │
│       ▼                                                              │
│  ZImagePipeline.init()                                              │
│       │                                                              │
│       ├──► Dit.init() ──► ManagedMLModel (NOT loaded yet)           │
│       │                                                              │
│       └──► AutoencoderKLZImage.init() ──► ManagedMLModel (NOT loaded)│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Generation Time                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ZImagePipeline.generateImages()                                    │
│       │                                                              │
│       ├──► loadEmbeddings() - Load text embeddings (~770KB)         │
│       │                                                              │
│       ├──► generateLatentSample() - Create random noise (~256KB)    │
│       │                                                              │
│       │   ┌─────── Denoising Loop (4 steps) ───────┐                │
│       │   │                                         │                │
│       ├──►│  dit.predictNoise()                     │                │
│       │   │       │                                 │                │
│       │   │       └──► ManagedMLModel.perform()     │                │
│       │   │               │                         │                │
│       │   │               └──► MLModel load (FIRST TIME ONLY)       │
│       │   │                                         │                │
│       │   │  scheduler.step()                       │                │
│       │   │                                         │                │
│       │   └─────────────────────────────────────────┘                │
│       │                                                              │
│       ├──► dit.unloadResources() (if reduceMemory=true)             │
│       │                                                              │
│       └──► vae.decode()                                             │
│               │                                                      │
│               └──► ManagedMLModel.perform() ──► MLModel load        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Memory Management

### ManagedMLModel

Each CoreML model is wrapped in `ManagedMLModel` which provides:

- **Lazy Loading**: Models load on first use (`perform()` call)
- **Thread Safety**: Serial dispatch queue protects model access
- **Autoreleasepool**: Wraps predictions to release temporary allocations
- **Explicit Unload**: `unloadResources()` sets model reference to nil

### reduceMemory Flag

When `reduceMemory = true` (default for mobile):

1. **Prewarm only**: `loadResources()` precompiles models but doesn't keep them loaded
2. **Sequential execution**: DiT is unloaded before VAE loads
3. **Automatic cleanup**: Models unload after their task completes

### Memory Timeline (reduceMemory=true)

```
Memory Usage
    ^
    │
500 │              ┌───────┐
    │              │  DiT  │
400 │              │       │
    │              │       │
300 │              │       │              ┌───────┐
    │              │       │              │  VAE  │
200 │              │       │              │       │
    │   ┌──────┐   │       │              │       │
100 │   │ Base │   │       │              │       │
    │   │      │   │       │              │       │
  0 │───┴──────┴───┴───────┴──────────────┴───────┴──► Time
    │   App      DiT      DiT            VAE     VAE
    │  Start    Load    Unload          Load   Unload
```

### Memory Timeline (reduceMemory=false) - PROBLEMATIC

```
Memory Usage
    ^
    │
700 │                        ┌─────────────────────┐
    │              ┌─────────┤  DiT + VAE          │
600 │              │         │                     │
    │              │  DiT    │                     │
500 │              │         │                     │
    │              │         │                     │
400 │              │         │                     │
    │              │         │                     │
300 │              │         │                     │
    │   ┌──────┐   │         │                     │
100 │   │ Base │   │         │                     │
    │   │      │   │         │                     │
  0 │───┴──────┴───┴─────────┴─────────────────────┴──► Time
    │   App      Both       ← MEMORY PRESSURE HERE
    │  Start    Load
```

## Compute Units

The pipeline supports different compute unit configurations:

| Configuration | Description | Memory Location |
|--------------|-------------|-----------------|
| `.cpuAndNeuralEngine` | Uses ANE + CPU (default) | ANE has dedicated memory |
| `.cpuAndGPU` | Uses GPU + CPU | Shared system memory |
| `.all` | Uses best available | May choose GPU |
| `.cpuOnly` | CPU only | System memory |

**Recommendation**: Use `.cpuAndNeuralEngine` on iOS devices to leverage the Neural Engine's dedicated memory pool.

## Model Files

Expected model files in the resources directory:

```
Resources/
├── ZImageTurbo_TransformerBackbone_fp16.mlmodelc/   # DiT model (~300-500MB)
├── VAEDecoder.mlmodelc/                              # VAE decoder (~80-150MB)
└── zimage_embeddings.bin                             # Pre-computed embeddings (~770KB)
```

## Debugging Memory Issues

The pipeline includes memory logging at key checkpoints. Enable by building in Debug mode:

```
[Memory] ZImagePipeline.loadResources.start: 150.2 MB
[Memory] ManagedMLModel[ZImageTurbo_TransformerBackbone_fp16.mlmodelc].load.before: 150.3 MB
[ManagedMLModel] Loading model: ZImageTurbo_TransformerBackbone_fp16.mlmodelc
[ManagedMLModel] Loaded ZImageTurbo_TransformerBackbone_fp16.mlmodelc in 2.35s
[Memory] ManagedMLModel[ZImageTurbo_TransformerBackbone_fp16.mlmodelc].load.after: 523.7 MB
```

### Memory Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `PipelineLoader.loadAppPipeline.start` | App pipeline loading begins |
| `ZImagePipeline.loadResources.beforeDit` | Before DiT model loads |
| `ZImagePipeline.loadResources.afterDit` | After DiT model loads |
| `ZImagePipeline.generateImages.afterDitUnload` | After DiT unloads (reduceMemory) |
| `ZImagePipeline.generateImages.beforeDecode` | Before VAE loads |
| `ZImagePipeline.generateImages.afterDecode` | After VAE decoding completes |

## Troubleshooting

### App Crashes on Launch

1. Check memory logs for spike before crash
2. Verify `reduceMemory = true` is set
3. Ensure compute units are `.cpuAndNeuralEngine`
4. Check model file sizes match expected values

### "Exited Unexpectedly" (Jetsam Kill)

iOS terminates apps exceeding memory limits. Solutions:

1. Ensure sequential model loading (DiT unloads before VAE loads)
2. Use FP16 models instead of FP32
3. Consider INT8 quantized models for extreme constraints

### Generation Produces NaN Values

1. Verify model input dtypes match model spec (FP16 vs FP32)
2. Check that FP16 conversion happens at model boundary
3. Ensure embeddings file is not corrupted
