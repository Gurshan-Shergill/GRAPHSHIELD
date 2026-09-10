import torch
import open_clip
from PIL import Image
from typing import Optional
import numpy as np
import logging

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_clip_model = None
_clip_preprocess = None
_tokenizer = None


def get_clip_model():
    '''Load CLIP model (singleton)'''
    global _clip_model, _clip_preprocess, _tokenizer
    if _clip_model is None:
        try:
            logger.info(f'Loading CLIP model: {settings.clip_model} ({settings.clip_pretrained})')
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                settings.clip_model,
                pretrained=settings.clip_pretrained,
                device='cpu',
            )
            _tokenizer = open_clip.get_tokenizer(settings.clip_model)
            _clip_model.eval()
            logger.info('CLIP model loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load CLIP model: {e}')
            raise
    return _clip_model, _clip_preprocess, _tokenizer


def get_clip_embedding(image: Image.Image) -> np.ndarray:
    '''Generate CLIP embedding for an image'''
    model, preprocess, _ = get_clip_model()

    # Preprocess image
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).cpu().numpy().astype(np.float32)


def get_text_embedding(text: str) -> np.ndarray:
    '''Generate CLIP embedding for text'''
    model, _, tokenizer = get_clip_model()

    text_input = tokenizer([text])

    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.squeeze(0).cpu().numpy().astype(np.float32)
