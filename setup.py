"""
ALSSSP - Adaptive Learning Single-Source Shortest Path
======================================================

Also known as Turbo-Dijkstra - A high-performance shortest path library
achieving 10-50x speedup over standard Dijkstra for point-to-point queries.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="turbo-dijkstra",
    version="1.0.0",
    author="Dr. Ankush Rai",
    author_email="ankushrai@bitdurg.ac.in",
    description="ALSSSP: Adaptive Learning Single-Source Shortest Path - 50x faster Dijkstra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RaiAnk/turbo-dijkstra",
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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
        "pathfinding",
        "optimization",
        "routing",
        "navigation",
        "turbo-dijkstra",
        "alsssp",
    ],
    project_urls={
        "Documentation": "https://github.com/RaiAnk/turbo-dijkstra#readme",
        "Bug Reports": "https://github.com/RaiAnk/turbo-dijkstra/issues",
        "Source": "https://github.com/RaiAnk/turbo-dijkstra",
        "YouTube": "https://youtu.be/HHgXiHJqJDU",
    },
)
