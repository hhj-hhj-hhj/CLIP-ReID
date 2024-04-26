import torch
import torch.nn as nn
from .clip import clip


class IMG2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

def get_text_features(model, token_feature, clip_model, dtype):
    b = token_feature.size(0)
    text_tokenize = clip.tokenize('A photo of X')
    text = text_tokenize.view(1, -1)
    text = text.repeat(token_feature.size(0), 1)
    with torch.no_grad():
        text_embedding = clip_model.token_embedding(text).type(dtype)
    prompts = torch.cat((text_embedding[:, :-1, :], token_feature), dim=1)
    text_features = model.text_encoder(prompts, text_tokenize)