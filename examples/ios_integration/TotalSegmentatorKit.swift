import Foundation
import CoreML
import Vision
import Accelerate

/// TotalSegmentator inference manager for iOS
public class TotalSegmentatorKit {
    
    // MARK: - Properties
    
    private var bodyModel: MLModel?
    private var lungVesselsModel: MLModel?
    private var cerebralBleedModel: MLModel?
    private var hipImplantModel: MLModel?
    private var coronaryArteriesModel: MLModel?
    
    private let processingQueue = DispatchQueue(label: "com.totalsegmentator.processing", qos: .userInitiated)
    
    // MARK: - Model Types
    
    public enum ModelType: String, CaseIterable {
        case body = "body"
        case lungVessels = "lung_vessels"
        case cerebralBleed = "cerebral_bleed"
        case hipImplant = "hip_implant"
        case coronaryArteries = "coronary_arteries"
        
        var className: String {
            switch self {
            case .body: return "TotalSegmentatorBody"
            case .lungVessels: return "TotalSegmentatorLungVessels"
            case .cerebralBleed: return "TotalSegmentatorCerebralBleed"
            case .hipImplant: return "TotalSegmentatorHipImplant"
            case .coronaryArteries: return "TotalSegmentatorCoronaryArteries"
            }
        }
        
        var numClasses: Int {
            switch self {
            case .body: return 104
            case .lungVessels: return 6
            case .cerebralBleed: return 4
            case .hipImplant: return 2
            case .coronaryArteries: return 3
            }
        }
    }
    
    // MARK: - Initialization
    
    public init() {}
    
    /// Load a specific model
    public func loadModel(_ type: ModelType) throws {
        guard let modelURL = Bundle.main.url(forResource: type.rawValue, withExtension: "mlmodelc") else {
            throw SegmentationError.modelNotFound(type.rawValue)
        }
        
        let model = try MLModel(contentsOf: modelURL)
        
        switch type {
        case .body: bodyModel = model
        case .lungVessels: lungVesselsModel = model
        case .cerebralBleed: cerebralBleedModel = model
        case .hipImplant: hipImplantModel = model
        case .coronaryArteries: coronaryArteriesModel = model
        }
    }
    
    /// Load all available models
    public func loadAllModels() throws {
        for modelType in ModelType.allCases {
            do {
                try loadModel(modelType)
                print("Loaded \(modelType.rawValue) model")
            } catch {
                print("Failed to load \(modelType.rawValue): \(error)")
            }
        }
    }
    
    // MARK: - Inference
    
    /// Perform segmentation on a CT volume
    public func segment(
        volume: MLMultiArray,
        modelType: ModelType,
        completion: @escaping (Result<SegmentationResult, Error>) -> Void
    ) {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                let result = try self.performSegmentation(volume: volume, modelType: modelType)
                DispatchQueue.main.async {
                    completion(.success(result))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    private func performSegmentation(
        volume: MLMultiArray,
        modelType: ModelType
    ) throws -> SegmentationResult {
        
        // Get the appropriate model
        let model: MLModel
        switch modelType {
        case .body:
            guard let m = bodyModel else { throw SegmentationError.modelNotLoaded(modelType.rawValue) }
            model = m
        case .lungVessels:
            guard let m = lungVesselsModel else { throw SegmentationError.modelNotLoaded(modelType.rawValue) }
            model = m
        case .cerebralBleed:
            guard let m = cerebralBleedModel else { throw SegmentationError.modelNotLoaded(modelType.rawValue) }
            model = m
        case .hipImplant:
            guard let m = hipImplantModel else { throw SegmentationError.modelNotLoaded(modelType.rawValue) }
            model = m
        case .coronaryArteries:
            guard let m = coronaryArteriesModel else { throw SegmentationError.modelNotLoaded(modelType.rawValue) }
            model = m
        }
        
        // Prepare input
        let input = try MLDictionaryFeatureProvider(dictionary: ["volume": volume])
        
        // Run inference
        let startTime = Date()
        let output = try model.prediction(from: input)
        let inferenceTime = Date().timeIntervalSince(startTime)
        
        // Extract segmentation
        guard let segmentation = output.featureValue(for: "segmentation")?.multiArrayValue else {
            throw SegmentationError.invalidOutput
        }
        
        // Create result
        return SegmentationResult(
            segmentation: segmentation,
            modelType: modelType,
            inferenceTime: inferenceTime
        )
    }
    
    // MARK: - Preprocessing
    
    /// Preprocess CT volume for model input
    public func preprocessVolume(
        _ volume: [[[Float]]],
        targetSpacing: (Float, Float, Float) = (1.5, 1.5, 1.5)
    ) throws -> MLMultiArray {
        
        let depth = volume.count
        let height = volume[0].count
        let width = volume[0][0].count
        
        // Create MLMultiArray
        let shape = [1, 1, depth, height, width] as [NSNumber]
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            throw SegmentationError.preprocessingFailed
        }
        
        // Fill array with normalized values
        var index = 0
        for d in 0..<depth {
            for h in 0..<height {
                for w in 0..<width {
                    // Apply HU windowing and normalization
                    let huValue = volume[d][h][w]
                    let clipped = max(-1000, min(1000, huValue))
                    let normalized = (clipped + 1000) / 2000
                    
                    multiArray[index] = NSNumber(value: normalized)
                    index += 1
                }
            }
        }
        
        return multiArray
    }
    
    // MARK: - Postprocessing
    
    /// Convert segmentation to label map
    public func extractLabelMap(
        from segmentation: MLMultiArray,
        threshold: Float = 0.5
    ) -> [[[Int]]] {
        
        // Get dimensions (assuming 5D: batch, channel, depth, height, width)
        let depth = segmentation.shape[2].intValue
        let height = segmentation.shape[3].intValue
        let width = segmentation.shape[4].intValue
        
        var labelMap = Array(repeating: 
            Array(repeating: 
                Array(repeating: 0, count: width), 
                count: height), 
            count: depth)
        
        // Extract predictions
        var index = 0
        for d in 0..<depth {
            for h in 0..<height {
                for w in 0..<width {
                    let value = segmentation[index].floatValue
                    labelMap[d][h][w] = value > threshold ? 1 : 0
                    index += 1
                }
            }
        }
        
        return labelMap
    }
    
    /// Calculate volume statistics
    public func calculateVolumeStats(
        labelMap: [[[Int]]],
        spacing: (Float, Float, Float) = (1.5, 1.5, 1.5)
    ) -> VolumeStatistics {
        
        var voxelCount = 0
        for slice in labelMap {
            for row in slice {
                for voxel in row {
                    if voxel > 0 {
                        voxelCount += 1
                    }
                }
            }
        }
        
        let voxelVolume = spacing.0 * spacing.1 * spacing.2
        let totalVolume = Float(voxelCount) * voxelVolume
        
        return VolumeStatistics(
            voxelCount: voxelCount,
            volumeMM3: totalVolume,
            volumeML: totalVolume / 1000
        )
    }
}

// MARK: - Result Types

public struct SegmentationResult {
    public let segmentation: MLMultiArray
    public let modelType: TotalSegmentatorKit.ModelType
    public let inferenceTime: TimeInterval
}

public struct VolumeStatistics {
    public let voxelCount: Int
    public let volumeMM3: Float
    public let volumeML: Float
}

// MARK: - Errors

public enum SegmentationError: LocalizedError {
    case modelNotFound(String)
    case modelNotLoaded(String)
    case invalidInput
    case invalidOutput
    case preprocessingFailed
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .modelNotLoaded(let name):
            return "Model not loaded: \(name)"
        case .invalidInput:
            return "Invalid input format"
        case .invalidOutput:
            return "Invalid model output"
        case .preprocessingFailed:
            return "Failed to preprocess volume"
        }
    }
}

// MARK: - Extensions

extension MLMultiArray {
    /// Convenience initializer for 3D medical images
    public convenience init?(shape: [Int], dataType: MLMultiArrayDataType = .float32) throws {
        let nsShape = shape.map { NSNumber(value: $0) }
        try self.init(shape: nsShape, dataType: dataType)
    }
}