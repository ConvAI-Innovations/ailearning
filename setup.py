#!/usr/bin/env python3
"""
Setup script for ConvAI Innovations.

This file is mainly for compatibility with older build tools.
The primary configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from pyproject.toml or define them here
install_requires = [
    "torch>=1.13.0",
    "numpy>=1.21.0",
    "llama-cpp-python>=0.2.0",
    "requests>=2.25.0",
]

extras_require = {
    "audio": [
        "kokoro-tts>=0.1.0",
        "sounddevice>=0.4.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
        "build>=0.10.0",
        "twine>=4.0.0",
    ],
}

# Add 'all' extra that includes everything
extras_require["all"] = [
    req for extra in extras_require.values() for req in extra
]

setup(
    name="convai-innovations",
    version="1.0.0",
    author="ConvAI Innovations",
    author_email="contact@convai-innovations.com",
    description="Interactive LLM Training Academy - Learn to build language models from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/convai-innovations/convai-innovations",
    project_urls={
        "Bug Tracker": "https://github.com/convai-innovations/convai-innovations/issues",
        "Documentation": "https://convai-innovations.readthedocs.io/",
        "Source Code": "https://github.com/convai-innovations/convai-innovations",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "convai=convai_innovations.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning", 
        "deep-learning", 
        "llm", 
        "education", 
        "pytorch", 
        "transformer", 
        "ai-training",
        "interactive-learning",
        "neural-networks"
    ],
)