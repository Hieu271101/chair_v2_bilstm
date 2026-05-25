import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    from torchvision.models import Inception_V3_Weights
except Exception:
    Inception_V3_Weights = None


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP between AvgPool and MaxPool per CBAM
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        y = avg_out + max_out
        return self.sigmoid(y).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class BaseModel(nn.Module):
    def __init__(self, embed_dim: int = 64, pretrained: bool = True):
        super().__init__()
        # Use InceptionV3 backbone per paper (keep output channels 2048)
        if Inception_V3_Weights is not None:
            weights = Inception_V3_Weights.DEFAULT if pretrained else None
            inception = models.inception_v3(weights=weights)
        else:
            inception = models.inception_v3(pretrained=pretrained)
            
        if hasattr(inception, 'aux_logits'):
            inception.aux_logits = False

        # Extract features up to Mixed_7c (output channels 2048)
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.maxpool1,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.maxpool2,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )
        self.channel_att = ChannelAttention(in_channels=2048)
        self.spatial_att = SpatialAttention()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, embed_dim)

    def feature_extractor(self, x):
        B = self.features(x)
        # Apply CBAM: channel attention multiplicative, then spatial multiplicative
        att_c = self.channel_att(B)
        B_c = B * att_c
        att_s = self.spatial_att(B_c)
        B_out = B_c * att_s
        # Residual connection per paper (Eq. 2): add back the original features
        B_res = B + B_out
        pooled = self.global_pool(B_res).flatten(1)
        return pooled

    def forward(self, x):
        pooled = self.feature_extractor(x)
        emb = self.fc(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class BiLSTMModule(nn.Module):
    # Paper defaults: 2 layers, hidden_dim=512 (bidirectional -> 1024 before FC)
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=True)   # bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sequence):
        out, _ = self.lstm(sequence)          # (B, T, hidden_dim * 2)
        embedded = self.fc(out)               # (B, T, output_dim)
        embedded = F.normalize(embedded, p=2, dim=2)
        return embedded


def train_base_model(base_model: nn.Module, dataloader, device=None, epochs: int = 40, lr: float = 0.01):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    base_model = base_model.to(device)
    triplet_loss = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adagrad(base_model.parameters(), lr=lr) # Use Adagrad per paper
    base_model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for sketch, pos_photo, neg_photo in dataloader:
            sketch = sketch.to(device)
            pos_photo = pos_photo.to(device)
            neg_photo = neg_photo.to(device)
            
            emb_s = base_model(sketch)
            emb_p = base_model(pos_photo)
            emb_n = base_model(neg_photo)
            
            loss = triplet_loss(emb_s, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / (len(dataloader) or 1)
        print(f"Epoch {epoch+1} loss: {avg:.4f}")
        torch.save(base_model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


def evaluate_embeddings(gallery_embeddings: dict, query_embeddings: list, true_indices: list, k_list=(1, 5, 10)):
    gallery_ids = list(gallery_embeddings.keys())
    gallery_tensor = torch.stack([gallery_embeddings[pid] for pid in gallery_ids])
    query_tensor = torch.stack(query_embeddings)

    device = query_tensor.device
    gallery_tensor = gallery_tensor.to(device)

    dist_matrix = 2.0 * (1.0 - torch.mm(query_tensor, gallery_tensor.t()))

    results = {}
    for k in k_list:
        _, top_k_indices = torch.topk(dist_matrix, k, dim=1, largest=False)
        correct = 0
        for q_idx, true_id in enumerate(true_indices):
            top_ids = [gallery_ids[idx] for idx in top_k_indices[q_idx].tolist()]
            if true_id in top_ids:
                correct += 1
        results[f'P@{k}'] = correct / len(query_embeddings)
    return results


__all__ = [
    'TripletLoss', 'BiLSTMModule', 'BaseModel', 'train_base_model', 'evaluate_embeddings'
]
