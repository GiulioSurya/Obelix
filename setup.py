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
    description="A multi-provider LLM agent framework with tool support and middleware system",
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
        "pandas>=2.2.0",
        "numpy>=2.4.0",
        "pydantic>=2.12.0",

        # Core - Logging
        "loguru",

        # Core - Configuration
        "python-dotenv>=1.2.0",
        "pyyaml",

        # Core - Utilities
        "httpx>=0.28.0",
        "docstring_parser>=0.17.0",
        "circuitbreaker>=2.1.0",
        "cachetools>=6.0.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        # LLM Providers (optional)
        "anthropic": ["anthropic>=0.75.0"],
        "oci": ["oci>=2.164.0"],
        "ibm": ["ibm_watsonx_ai>=1.4.0"],
        "ollama": ["ollama>=0.6.0"],

        # Database Connections (optional)
        "oracle": ["oracledb>=2.0.0"],
        "postgres": ["psycopg[binary,pool]>=3.0.0"],

        # Development
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
        ],

        # All LLM providers
        "all-llm": [
            "anthropic>=0.75.0",
            "oci>=2.164.0",
            "ibm_watsonx_ai>=1.4.0",
            "ollama>=0.6.0",
        ],

        # All databases
        "all-db": [
            "oracledb>=2.0.0",
            "psycopg[binary,pool]>=3.0.0",
        ],

        # Everything
        "all": [
            "anthropic>=0.75.0",
            "oci>=2.164.0",
            "ibm_watsonx_ai>=1.4.0",
            "ollama>=0.6.0",
            "oracledb>=2.0.0",
            "psycopg[binary,pool]>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "obelix=src.main:main",
        ],
    },
)
