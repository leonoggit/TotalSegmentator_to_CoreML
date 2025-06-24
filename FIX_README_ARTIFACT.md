# Fix: README.md Missing from Source Artifact

## Problem
The convert-to-coreml job failed during `pip install -e .` with:
```
FileNotFoundError: [Errno 2] No such file or directory: '/Users/runner/work/TotalSegmentator_to_CoreML/TotalSegmentator_to_CoreML/README.md'
```

## Root Cause
The source-code artifact uploaded in the prepare-models job didn't include README.md, but setup.py tries to read it during installation:

```python
# setup.py line 10-11
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")
```

## Solution
Added README.md to the source-code artifact upload:

```yaml
- name: Upload source code
  uses: actions/upload-artifact@v4
  with:
    name: source-code
    path: |
      src/
      scripts/
      convert_totalsegmentator.py
      requirements.txt
      setup.py
      README.md  # ← Added this line
```

## Additional Improvements
Also added missing dependencies to the installation step to ensure all required packages are available:
- nibabel (medical imaging)
- SimpleITK (medical imaging)
- scipy (scientific computing)
- matplotlib (visualization)
- pandas (data handling)
- Pillow (image processing)

## Verification
The workflow should now:
1. ✅ Upload README.md with source code
2. ✅ Successfully install the package with `pip install -e .`
3. ✅ Continue to the conversion step

## Lesson Learned
When using artifacts to transfer files between jobs, ensure ALL files referenced by setup.py or other installation scripts are included in the artifact.