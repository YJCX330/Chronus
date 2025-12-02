import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor,WhisperModel
import chronus.model.speech_encoder.whisper as whisper
from chronus.model.speech_encoder.beats.BEATs import BEATsConfig, BEATs

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from chronus.model.speech_encoder.whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)
        try:
            encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
            replace_layer_norm(encoder)
        except:
            encoder=None
        return encoder
    
class DualWrappedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.whisper_model = self.load_whisper(config)
        self.beats_model = self.load_beats(config)
    
    def load_whisper(cls, model_config):

        def replace_layer_norm(module):
            from chronus.model.speech_encoder.whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)
        try:
            encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
            replace_layer_norm(encoder)
        except:
            encoder=None
        return encoder

    def load_beats(cls, model_config):
        try:
            beats_path = model_config.music_encoder
            print("Loading BEATs Model: ",model_config.music_encoder)
            beats_ckpt = torch.load(beats_path, map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            beats = BEATs(beats_cfg)
            beats.load_state_dict(beats_ckpt['model'])
        except: 
            beats=None
        return beats

    def forward(self, x, raw_wav=None, audio_padding_mask=None):
        with torch.no_grad():
            speech_embeds = self.whisper_model(x)
            audio_embeds, _ = self.beats_model.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
        if audio_embeds.size(1) < speech_embeds.size(1):
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        elif audio_embeds.size(1) > speech_embeds.size(1):
            speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
        speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
        speech_embeds = speech_embeds.to(torch.bfloat16)
        return speech_embeds