import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


class MLPProjector(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.llm_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size),
        )

    def forward(self, x):
        return self.projector(x)
