#!/usr/bin/env python3
"""
Download TotalSegmentator PyTorch models
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib
import json


# Model URLs and checksums (placeholder - replace with actual URLs)
MODEL_INFO = {
    "body": {
        "url": "https://example.com/totalsegmentator/body.pth",
        "sha256": "placeholder_hash",
        "size_mb": 187
    },
    "lung_vessels": {
        "url": "https://example.com/totalsegmentator/lung_vessels.pth",
        "sha256": "placeholder_hash",
        "size_mb": 156
    },
    "cerebral_bleed": {
        "url": "https://example.com/totalsegmentator/cerebral_bleed.pth",
        "sha256": "placeholder_hash",
        "size_mb": 143
    },
    "hip_implant": {
        "url": "https://example.com/totalsegmentator/hip_implant.pth",
        "sha256": "placeholder_hash",
        "size_mb": 134
    },
    "coronary_arteries": {
        "url": "https://example.com/totalsegmentator/coronary_arteries.pth",
        "sha256": "placeholder_hash",
        "size_mb": 168
    }
}


def download_file(url: str, output_path: Path, expected_size_mb: int) -> bool:
    """Download file with progress bar"""
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
        
    except Exception as e:
        print(f"Error downloading {output_path.name}: {e}")
        return False


def verify_checksum(file_path: Path, expected_hash: str) -> bool:
    """Verify file checksum"""
    
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    calculated_hash = sha256_hash.hexdigest()
    return calculated_hash == expected_hash


def download_models(output_dir: Path, 
                   models: list = None,
                   token: str = None,
                   verify: bool = True) -> dict:
    """Download TotalSegmentator models"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if models is None:
        models = list(MODEL_INFO.keys())
    
    results = {}
    
    print(f"Downloading {len(models)} models to {output_dir}")
    
    for model_name in models:
        if model_name not in MODEL_INFO:
            print(f"Unknown model: {model_name}")
            results[model_name] = False
            continue
        
        model_info = MODEL_INFO[model_name]
        output_path = output_dir / f"{model_name}.pth"
        
        # Skip if already exists
        if output_path.exists():
            print(f"{model_name}.pth already exists, skipping...")
            results[model_name] = True
            continue
        
        # Download
        print(f"\nDownloading {model_name} ({model_info['size_mb']} MB)...")
        
        # Add token to URL if provided
        url = model_info['url']
        if token:
            url += f"?token={token}"
        
        success = download_file(url, output_path, model_info['size_mb'])
        
        if success and verify and model_info['sha256'] != 'placeholder_hash':
            print(f"Verifying checksum...")
            if not verify_checksum(output_path, model_info['sha256']):
                print(f"Checksum verification failed for {model_name}")
                output_path.unlink()
                success = False
        
        results[model_name] = success
    
    # Summary
    successful = sum(results.values())
    print(f"\nDownload complete: {successful}/{len(models)} successful")
    
    # Save download log
    log_path = output_dir / "download_log.json"
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download TotalSegmentator PyTorch models"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/pytorch",
        help="Output directory for models"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_INFO.keys()) + ["all"],
        default=["all"],
        help="Models to download"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="Authentication token (if required)"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification"
    )
    
    args = parser.parse_args()
    
    # Parse models
    if "all" in args.models:
        models = None
    else:
        models = args.models
    
    # Download
    output_dir = Path(args.output_dir)
    results = download_models(
        output_dir,
        models=models,
        token=args.token,
        verify=not args.no_verify
    )
    
    # Exit with error if any download failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    # Check if running in GitHub Actions
    if os.getenv("GITHUB_ACTIONS"):
        print("Note: Running in GitHub Actions")
        print("Make sure TOTALSEGMENTATOR_TOKEN secret is set for authentication")
    
    main()