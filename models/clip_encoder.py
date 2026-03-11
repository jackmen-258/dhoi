"""
clip_encoder.py
===============
CLIP 文本编码器封装 (参数冻结, 懒加载)

用法:
    from models.clip_encoder import CLIPTextEncoder

    encoder = CLIPTextEncoder(model_name="ViT-B/32", device="cuda")
    text_feat = encoder.encode(["grasp the handle"])  # (1, 77, 512)
"""

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


class CLIPTextEncoder:
    """
    CLIP 文本编码器 (冻结参数, 懒加载)

    encode(texts) → (B, 77, 512) token-level 特征
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            import clip
            self._model, _ = clip.load(self.model_name, device=self.device)
            self._model = self._model.float()
            self._model.eval()
            self._tokenizer = clip.tokenize
            logger.info(f"[CLIP] Loaded {self.model_name} on {self.device}")
        except ImportError:
            logger.warning(
                "[CLIP] clip 未安装，文本条件将使用零向量。"
                "安装: pip install git+https://github.com/openai/CLIP.git"
            )
            self._model = None

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """texts → (B, 77, 512) token-level CLIP features"""
        self._lazy_load()
        if self._model is None:
            return torch.zeros(len(texts), 77, 512, device=self.device)

        tokens = self._tokenizer(texts, truncate=True).to(self.device)
        x = self._model.token_embedding(tokens)
        x = x + self._model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self._model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self._model.ln_final(x)
        return x.float()