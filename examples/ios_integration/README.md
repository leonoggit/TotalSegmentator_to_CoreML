# TotalSegmentator iOS Integration

This example demonstrates how to integrate TotalSegmentator CoreML models into an iOS application.

## Features

- **TotalSegmentatorKit**: Swift framework for easy model integration
- **3D Visualization**: Metal-based rendering of segmentation results
- **Multi-Model Support**: Switch between different segmentation models
- **Performance Monitoring**: Track inference time and memory usage
- **Volume Statistics**: Calculate segmented volume in mL and mm³

## Requirements

- iOS 15.0+
- Xcode 14.0+
- Swift 5.7+
- Device with A12 Bionic chip or later (for Neural Engine)

## Setup

1. **Add CoreML Models**
   ```bash
   # Copy converted models to the Xcode project
   cp ../../models/coreml/*.mlpackage TotalSegmentatorDemo/Models/
   ```

2. **Open Xcode Project**
   ```bash
   open TotalSegmentatorDemo.xcodeproj
   ```

3. **Configure Signing**
   - Select your development team
   - Update bundle identifier

4. **Build and Run**
   - Select target device (iPhone/iPad with Neural Engine recommended)
   - Build and run (⌘R)

## Usage

### Basic Integration

```swift
import TotalSegmentatorKit

// Initialize
let segmentator = TotalSegmentatorKit()

// Load models
try segmentator.loadModel(.body)

// Prepare input volume
let volume = try segmentator.preprocessVolume(ctData)

// Run segmentation
segmentator.segment(volume: volume, modelType: .body) { result in
    switch result {
    case .success(let segmentation):
        // Process results
        let labelMap = segmentator.extractLabelMap(from: segmentation.segmentation)
        let stats = segmentator.calculateVolumeStats(labelMap: labelMap)
        print("Segmented volume: \(stats.volumeML) mL")
        
    case .failure(let error):
        print("Segmentation failed: \(error)")
    }
}
```

### Advanced Features

#### Custom Preprocessing

```swift
// Custom HU windowing
let preprocessor = MedicalImagePreprocessor()
preprocessor.windowCenter = 40
preprocessor.windowWidth = 400
let processedVolume = preprocessor.process(ctVolume)
```

#### Batch Processing

```swift
// Process multiple volumes
let volumes = loadCTScans()
let results = try await segmentator.batchSegment(
    volumes: volumes,
    modelType: .body,
    maxConcurrent: 2
)
```

#### Memory-Efficient Processing

```swift
// Process large volumes in chunks
let chunker = VolumeChunker(chunkSize: 128, overlap: 16)
let chunks = chunker.createChunks(from: largeVolume)

var results: [SegmentationResult] = []
for chunk in chunks {
    let result = try await segmentator.segment(
        volume: chunk.data,
        modelType: .body
    )
    results.append(result)
}

// Reconstruct full segmentation
let fullSegmentation = chunker.reconstruct(from: results)
```

## Architecture

### TotalSegmentatorKit

The main framework providing:
- Model loading and management
- Preprocessing pipeline
- Inference execution
- Postprocessing utilities

### Key Components

1. **ModelManager**: Handles loading and caching of CoreML models
2. **Preprocessor**: CT data preprocessing (windowing, normalization)
3. **InferenceEngine**: Manages model execution on Neural Engine
4. **Postprocessor**: Converts model outputs to usable formats
5. **VolumeRenderer**: Metal-based 3D visualization

## Performance Optimization

### Neural Engine Utilization

```swift
// Force Neural Engine execution
let config = MLModelConfiguration()
config.computeUnits = .all  // Uses Neural Engine when available
let model = try MLModel(contentsOf: modelURL, configuration: config)
```

### Memory Management

```swift
// Release models when not needed
segmentator.unloadModel(.body)

// Use autorelease pool for batch processing
autoreleasepool {
    for volume in volumes {
        let result = try segmentator.segment(volume: volume, modelType: .body)
        processResult(result)
    }
}
```

### Background Processing

```swift
// Process in background
Task(priority: .userInitiated) {
    let result = try await segmentator.segment(
        volume: volume,
        modelType: .body
    )
    
    await MainActor.run {
        updateUI(with: result)
    }
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce volume size
   - Process in chunks
   - Use lower precision models

2. **Slow Performance**
   - Ensure Neural Engine is being used
   - Check thermal state
   - Profile with Instruments

3. **Model Loading Fails**
   - Verify model is included in bundle
   - Check iOS deployment target
   - Validate model format

### Debugging

```swift
// Enable debug logging
TotalSegmentatorKit.enableDebugLogging = true

// Monitor performance
segmentator.performanceHandler = { metrics in
    print("Inference time: \(metrics.inferenceTime)s")
    print("Peak memory: \(metrics.peakMemoryMB) MB")
}
```

## Sample App Structure

```
TotalSegmentatorDemo/
├── App/
│   ├── AppDelegate.swift
│   └── SceneDelegate.swift
├── Models/
│   ├── body.mlpackage
│   ├── lung_vessels.mlpackage
│   └── ...
├── Views/
│   ├── SegmentationViewController.swift
│   ├── VolumeRenderer.swift
│   └── Main.storyboard
├── TotalSegmentatorKit/
│   ├── TotalSegmentatorKit.swift
│   ├── Preprocessing.swift
│   ├── Postprocessing.swift
│   └── Extensions.swift
└── Resources/
    ├── Info.plist
    └── Assets.xcassets
```

## Best Practices

1. **Model Selection**
   - Use appropriate model for the task
   - Consider model size vs accuracy trade-off
   - Cache frequently used models

2. **Data Handling**
   - Validate input data format
   - Handle edge cases (empty volumes, invalid spacing)
   - Implement proper error handling

3. **User Experience**
   - Show progress during processing
   - Provide meaningful error messages
   - Allow cancellation of long operations

4. **Privacy**
   - Process data on-device only
   - No network requests required
   - Comply with healthcare regulations

## License

MIT License - See LICENSE file for details