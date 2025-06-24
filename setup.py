#!/usr/bin/env python3
"""
Setup script for TotalSegmentator to CoreML converter
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = [
    line.strip() 
    for line in requirements_path.read_text().splitlines() 
    if line.strip() and not line.startswith("#")
]

setup(
    name="totalsegmentator-coreml",
    version="1.0.0",
    author="TotalSegmentator CoreML Team",
    description="Convert TotalSegmentator PyTorch models to CoreML for iOS deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TotalSegmentator_to_CoreML",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "totalsegmentator-convert=convert_totalsegmentator:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="medical-imaging, coreml, totalsegmentator, ios, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TotalSegmentator_to_CoreML/issues",
        "Source": "https://github.com/yourusername/TotalSegmentator_to_CoreML",
    },
)