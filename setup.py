"""
Alpha-GPT-KR Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alpha-gpt-kr",
    version="0.1.0",
    author="Alpha-GPT-KR Team",
    description="AI-powered Alpha Mining for Korean Stock Market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alpha-gpt-kr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "langchain>=0.1.0",
        "FinanceDataReader>=0.9.50",
        "pykrx>=1.0.40",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
        "web": [
            "streamlit>=1.30.0",
            "gradio>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alpha-gpt=alpha_gpt_kr.cli:main",
        ],
    },
)
