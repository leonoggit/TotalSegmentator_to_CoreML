# Fix: MKL-DNN Operator Not Supported Error

## Problem
All models failed to convert with the error:
```
PyTorch convert function for op 'to_mkldnn' not implemented.
Exporting the operator 'aten::to_mkldnn' to ONNX opset version 13 is not supported.
```

## Root Cause
The PyTorch models were using Intel MKL-DNN (Math Kernel Library for Deep Neural Networks) optimizations, which create operators that are not compatible with CoreML conversion.

## Solution

### 1. Force CPU Mode in Model Creation (scripts/create_dummy_models.py)
```python
# Ensure model is in CPU mode and not using MKL-DNN
model = model.cpu()
model.eval()

# Save with compatibility flag
torch.save(
    model.state_dict(), 
    output_path, 
    _use_new_zipfile_serialization=False
)
```

### 2. Disable MKL-DNN in Converter (src/converter.py)
```python
# Force CPU for conversion to avoid MKL-DNN operators
pytorch_model = pytorch_model.cpu()
example_input = example_input.cpu()

# Disable MKL-DNN if available
if hasattr(torch, '_C') and hasattr(torch._C, '_set_mkldnn_enabled'):
    torch._C._set_mkldnn_enabled(False)
```

### 3. Update Model Loading (convert_totalsegmentator.py)
```python
# Load weights
model.load_state_dict(state_dict)
# Force CPU mode to avoid MKL-DNN issues during conversion
model.cpu()
model.eval()
```

### 4. Create Example Input on CPU
```python
# Create realistic CT data on CPU to avoid device issues
data = torch.randn(shape, device='cpu') * 500
```

### 5. Environment Variables in Workflow
```yaml
# Set environment variable to disable MKL
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## Why This Works

1. **CPU Mode**: Forces models to use standard CPU operations instead of MKL-DNN optimized ones
2. **Disable MKL-DNN**: Explicitly disables MKL-DNN backend if available
3. **Compatibility Save**: Uses older serialization format that's more compatible
4. **Environment Variables**: Limits MKL thread usage to prevent optimization

## Expected Outcome

- Models will be traced without MKL-DNN operators
- CoreML conversion will proceed without the `to_mkldnn` error
- ONNX fallback will work if direct conversion still fails

## Alternative Approaches

If issues persist:
1. Use `torch.jit.script` instead of `torch.jit.trace`
2. Export to ONNX first, then convert ONNX to CoreML
3. Use older PyTorch version without MKL-DNN (e.g., 1.13.0)