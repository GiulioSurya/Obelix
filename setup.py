"""
Obelix - Multi-Provider LLM Agent Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="obelix",
    version="0.1.0",
    author="Obelix Contributors",
    description="A multi-provider LLM agent framework with tool support and hooks system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/obelix/obelix",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.13",
    install_requires=[
        # Core - Data & Validation
        "pydantic>=2.12.0",
        "numpy>=2.2.0",

        # Core - Logging
        "loguru>=0.7.0",

        # Core - Configuration
        "pyyaml>=6.0.0",
    ],
    extras_require={
        # LLM Providers (optional)
        "anthropic": ["anthropic>=0.71.0"],
        "openai": ["openai>=2.14.0"],
        "oci": ["oci>=2.164.0"],
        "ibm": ["ibm_watsonx_ai>=1.4.0"],
        "ollama": ["ollama>=0.6.0"],
        "vllm": ["vllm>=0.13.0"],

        # MCP (Model Context Protocol)
        "mcp": ["mcp>=1.25.0"],

        # Development
        "dev": [
            "pytest>=9.0.0",
            "pytest-asyncio>=1.0.0",
            "pytest-cov>=7.0.0",
        ],

        # All LLM providers
        "all-llm": [
            "anthropic>=0.71.0",
            "openai>=2.14.0",
            "oci>=2.164.0",
            "ibm_watsonx_ai>=1.4.0",
            "ollama>=0.6.0",
            "vllm>=0.13.0",
        ],

        # Everything (all providers + mcp)
        "all": [
            "anthropic>=0.71.0",
            "openai>=2.14.0",
            "oci>=2.164.0",
            "ibm_watsonx_ai>=1.4.0",
            "ollama>=0.6.0",
            "vllm>=0.13.0",
            "mcp>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "obelix=src.main:main",
        ],
    },
)
