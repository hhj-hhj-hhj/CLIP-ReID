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
    return text_features

def get_loss_img2text(model, img2text, images, loss_img, loss_txt, clip_model):

    device = "cuda"
    with torch.no_grad():
        image_features = model(img=images, get_image=True)
    token_features = img2text(image_features)
    text_features = get_text_features(model, token_features, clip_model, dtype=clip_model.dtype)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = clip_model.logit_scale.exp()
    logits_per_image = logit_scale.mean()

    # 这是不使用分布式的代码
    ground_truth = torch.arange(len(image_features)).long()
    ground_truth.to(device)

    # Image loss.
    logits_per_image = logit_scale * image_features @ text_features.t()
    loss_img_val = loss_img(logits_per_image, ground_truth)
    logits_per_text = logit_scale * text_features @ image_features.t()
    loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss

def get_loss(model, images, texts, loss_img, loss_txt):
    device = "cuda"

    pass