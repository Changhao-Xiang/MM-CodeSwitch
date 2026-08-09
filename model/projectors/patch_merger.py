import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


class PatchMerger(nn.Module):
    def __init__(self, config: PretrainedConfig, scale_factor=0.5):
        super().__init__()
        self.scale_factor = scale_factor
        in_features = int(config.vision_config.hidden_size / (scale_factor**2))
        out_features = config.llm_config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

    # copy from modeling_internvl_chat.py
    def pixel_shuffle(self, x):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * self.scale_factor), int(c / self.scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * self.scale_factor),
            int(w * self.scale_factor),
            int(c / (self.scale_factor * self.scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x):
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, -1)
        x = self.pixel_shuffle(x)
        x = self.projector(x)
        return x.reshape(x.shape[0], -1, x.shape[-1])
