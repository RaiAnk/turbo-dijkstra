"""
ALSSSP - Adaptive Learning Single-Source Shortest Path
======================================================

A high-performance shortest path library achieving 10-50x speedup
over standard Dijkstra for point-to-point queries.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alsssp",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Adaptive Learning Single-Source Shortest Path Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/alsssp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "experiments": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "pandas>=1.4.0",
            "networkx>=2.8.0",
        ],
    },
    keywords=[
        "shortest-path",
        "dijkstra",
        "graph-algorithms",
        "bidirectional-search",
        "optimization",
    ],
    project_urls={
        "Documentation": "https://github.com/example/alsssp#readme",
        "Bug Reports": "https://github.com/example/alsssp/issues",
        "Source": "https://github.com/example/alsssp",
    },
)
