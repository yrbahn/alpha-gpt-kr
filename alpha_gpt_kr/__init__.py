"""
Alpha-GPT-KR: 한국 증시용 AI 기반 알파 마이닝 시스템
arXiv 2308.00016 논문 구현
"""

__version__ = "0.1.0"
__author__ = "Alpha-GPT-KR Team"

from .core import AlphaGPT
from .mining.operators import AlphaOperators
from .data.krx_loader import KRXDataLoader

__all__ = [
    "AlphaGPT",
    "AlphaOperators",
    "KRXDataLoader",
]
