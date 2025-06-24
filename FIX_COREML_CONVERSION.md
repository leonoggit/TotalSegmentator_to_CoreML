# Fix: CoreML Conversion Errors

## Problems Identified

### 1. **TorchScript Tracing Warning**
```
TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect
if x.shape != skip.shape:
```
**Cause**: Dynamic shape comparison in UNet skip connections
**Solution**: Removed conditional and always calculate padding (padding by 0 has no effect)

### 2. **CoreML Conversion Error**
```
ERROR - Conversion failed for body: 'list' object has no attribute 'val'
```
**Cause**: The conversion pipeline wasn't using the TotalSegmentatorConverter class which has proper error handling and ONNX fallback
**Solution**: Updated to use the converter's `convert()` method instead of custom implementation

### 3. **Redundant Code**
- Pipeline had its own `_convert_to_coreml` method duplicating converter functionality
- Optimization and saving were done twice

## Changes Made

### 1. Fixed UNet Model (src/models.py)
```python
# Before: Dynamic shape comparison
if x.shape != skip.shape:
    # padding logic

# After: Always calculate padding for TorchScript compatibility
diff_d = skip.size(2) - x.size(2)
diff_h = skip.size(3) - x.size(3)
diff_w = skip.size(4) - x.size(4)

x = F.pad(x, [...])  # Padding by 0 has no effect
```

### 2. Updated Conversion Pipeline (convert_totalsegmentator.py)
```python
# Before: Custom tracing and conversion
traced_model = torch.jit.trace(pytorch_model, example_input)
coreml_model = self._convert_to_coreml(traced_model, config, model_name)

# After: Use converter with error handling
coreml_model = self.converter.convert(
    model_name=model_name,
    pytorch_model=pytorch_model,
    example_input=example_input,
    output_path=output_path,
    optimize=optimize
)
```

### 3. Removed Redundancy
- Removed duplicate optimization step (converter handles it)
- Removed duplicate save operation (converter handles it)
- Removed unused `_convert_to_coreml` method

## Benefits

1. **Better Error Handling**: The converter has try-except blocks with ONNX fallback
2. **Cleaner Code**: No duplicate functionality
3. **TorchScript Compatible**: Fixed dynamic shape comparisons
4. **More Robust**: Centralized conversion logic with proper error messages

## Expected Outcome

The conversion should now:
1. ✅ Trace models without warnings
2. ✅ Convert to CoreML with proper error handling
3. ✅ Fall back to ONNX if direct conversion fails
4. ✅ Save models with proper metadata

## Next Steps

1. Run the updated workflow to test the fixes
2. Monitor for any new errors
3. Verify converted models work correctly