import UIKit
import CoreML
import MetalKit
import simd

/// Example view controller demonstrating TotalSegmentator usage
class SegmentationViewController: UIViewController {
    
    // MARK: - UI Elements
    
    @IBOutlet weak var renderView: MTKView!
    @IBOutlet weak var modelSelector: UISegmentedControl!
    @IBOutlet weak var processButton: UIButton!
    @IBOutlet weak var statusLabel: UILabel!
    @IBOutlet weak var progressView: UIProgressView!
    @IBOutlet weak var statsTextView: UITextView!
    
    // MARK: - Properties
    
    private let segmentator = TotalSegmentatorKit()
    private var currentVolume: MLMultiArray?
    private var renderer: VolumeRenderer?
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupUI()
        setupRenderer()
        loadModels()
    }
    
    // MARK: - Setup
    
    private func setupUI() {
        processButton.layer.cornerRadius = 8
        statsTextView.layer.borderColor = UIColor.systemGray4.cgColor
        statsTextView.layer.borderWidth = 1
        statsTextView.layer.cornerRadius = 8
        
        // Configure model selector
        modelSelector.removeAllSegments()
        for (index, model) in TotalSegmentatorKit.ModelType.allCases.enumerated() {
            modelSelector.insertSegment(withTitle: model.rawValue.capitalized, at: index, animated: false)
        }
        modelSelector.selectedSegmentIndex = 0
    }
    
    private func setupRenderer() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            showAlert(title: "Error", message: "Metal is not supported on this device")
            return
        }
        
        renderView.device = device
        renderView.colorPixelFormat = .bgra8Unorm
        renderView.preferredFramesPerSecond = 30
        
        renderer = VolumeRenderer(device: device, view: renderView)
        renderView.delegate = renderer
    }
    
    private func loadModels() {
        updateStatus("Loading models...")
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                try self?.segmentator.loadAllModels()
                
                DispatchQueue.main.async {
                    self?.updateStatus("Models loaded successfully")
                    self?.processButton.isEnabled = true
                }
            } catch {
                DispatchQueue.main.async {
                    self?.showAlert(title: "Error", message: "Failed to load models: \(error.localizedDescription)")
                }
            }
        }
    }
    
    // MARK: - Actions
    
    @IBAction func processButtonTapped(_ sender: UIButton) {
        guard let volume = loadSampleVolume() else {
            showAlert(title: "Error", message: "Failed to load sample volume")
            return
        }
        
        currentVolume = volume
        performSegmentation()
    }
    
    @IBAction func modelSelectorChanged(_ sender: UISegmentedControl) {
        if currentVolume != nil {
            performSegmentation()
        }
    }
    
    // MARK: - Segmentation
    
    private func performSegmentation() {
        guard let volume = currentVolume else { return }
        
        let modelType = TotalSegmentatorKit.ModelType.allCases[modelSelector.selectedSegmentIndex]
        
        updateStatus("Processing with \(modelType.rawValue)...")
        processButton.isEnabled = false
        progressView.progress = 0
        progressView.isHidden = false
        
        // Animate progress
        UIView.animate(withDuration: 2.0) {
            self.progressView.setProgress(0.8, animated: true)
        }
        
        segmentator.segment(volume: volume, modelType: modelType) { [weak self] result in
            self?.progressView.setProgress(1.0, animated: true)
            
            switch result {
            case .success(let segmentationResult):
                self?.handleSegmentationResult(segmentationResult)
            case .failure(let error):
                self?.showAlert(title: "Segmentation Failed", message: error.localizedDescription)
            }
            
            self?.processButton.isEnabled = true
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self?.progressView.isHidden = true
            }
        }
    }
    
    private func handleSegmentationResult(_ result: SegmentationResult) {
        updateStatus("Segmentation completed in \(String(format: "%.2f", result.inferenceTime))s")
        
        // Extract label map
        let labelMap = segmentator.extractLabelMap(from: result.segmentation)
        
        // Calculate statistics
        let stats = segmentator.calculateVolumeStats(labelMap: labelMap)
        
        // Update UI
        updateStats(stats, modelType: result.modelType)
        
        // Update 3D visualization
        renderer?.updateSegmentation(labelMap)
    }
    
    // MARK: - UI Updates
    
    private func updateStatus(_ message: String) {
        DispatchQueue.main.async {
            self.statusLabel.text = message
        }
    }
    
    private func updateStats(_ stats: VolumeStatistics, modelType: TotalSegmentatorKit.ModelType) {
        let statsText = """
        Model: \(modelType.rawValue)
        Classes: \(modelType.numClasses)
        
        Segmentation Results:
        • Voxels: \(stats.voxelCount)
        • Volume: \(String(format: "%.1f", stats.volumeML)) mL
        • Volume: \(String(format: "%.1f", stats.volumeMM3)) mm³
        """
        
        statsTextView.text = statsText
    }
    
    // MARK: - Helpers
    
    private func loadSampleVolume() -> MLMultiArray? {
        // Load sample CT data
        // In a real app, this would load from DICOM files or other sources
        
        let shape = [128, 128, 128]
        var volume: [[[Float]]] = []
        
        // Generate synthetic CT data for demo
        for z in 0..<shape[0] {
            var slice: [[Float]] = []
            for y in 0..<shape[1] {
                var row: [Float] = []
                for x in 0..<shape[2] {
                    // Create a sphere in the center
                    let center = Float(shape[0] / 2)
                    let distance = sqrt(
                        pow(Float(x) - center, 2) +
                        pow(Float(y) - center, 2) +
                        pow(Float(z) - center, 2)
                    )
                    
                    let value: Float
                    if distance < 30 {
                        value = 200 // Soft tissue
                    } else if distance < 35 {
                        value = 1000 // Bone
                    } else {
                        value = -500 // Air/background
                    }
                    
                    row.append(value + Float.random(in: -50...50))
                }
                slice.append(row)
            }
            volume.append(slice)
        }
        
        // Preprocess
        return try? segmentator.preprocessVolume(volume)
    }
    
    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - Volume Renderer

class VolumeRenderer: NSObject, MTKViewDelegate {
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var segmentationData: [[[Int]]]?
    
    init(device: MTLDevice, view: MTKView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        super.init()
    }
    
    func updateSegmentation(_ labelMap: [[[Int]]]) {
        segmentationData = labelMap
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle view resize
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor else { return }
        
        // Clear to background color
        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        
        // In a real implementation, render the 3D segmentation here
        // This would involve creating meshes from the segmentation data
        // and rendering them with appropriate shaders
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}