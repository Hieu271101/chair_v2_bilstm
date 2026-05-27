"""
Train Bi-LSTM module on sequences of incomplete sketches (20 steps) with triplet loss.
The base model is frozen; only Bi-LSTM is trained.

Usage:
    python train_bilstm.py --triplets seq.csv --base-model chair_base.pth --epochs 500 --batch-size 20 --save bilstm_chair.pth
"""
import argparse
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import BaseModel, BiLSTMModule, TripletLoss


class SequenceTripletDataset(Dataset):
    """
    CSV format: each row: [json_list_of_step_paths, positive_photo_path, negative_photo_path]
    The json list contains exactly 20 paths (step1.png ... step20.png) in order.
    """
    def __init__(self, csv_path, transform=None):
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    self.rows.append(row)
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq_json, pos_path, neg_path = self.rows[idx]
        step_paths = json.loads(seq_json)   # list of 20 strings
        # Load all step images as tensors
        seq_tensors = []
        for p in step_paths:
            img = Image.open(p).convert('RGB')
            seq_tensors.append(self.transform(img))
        seq = torch.stack(seq_tensors, dim=0)   # (20, 3, 299, 299)
        pos = self.transform(Image.open(pos_path).convert('RGB'))
        neg = self.transform(Image.open(neg_path).convert('RGB'))
        return seq, pos, neg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triplets', required=True, help='CSV with sequence JSON, positive, negative')
    parser.add_argument('--base-model', required=True, help='Pretrained base model .pth')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=20, help='20 for Chair, 120 for Shoe')
    parser.add_argument('--lr', type=float, default=0.01, help='Adagrad learning rate')
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers for Bi-LSTM')
    parser.add_argument('--save', default='bilstm_model.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', default=None, help='Checkpoint file .pth để resume (chỉ Bi-LSTM)')
    parser.add_argument('--save-every', type=int, default=10, help='Lưu checkpoint mỗi N epoch')
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Transform (normalize to [-1, 1] for InceptionV3 feature extraction)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Dataset & loader
    dataset = SequenceTripletDataset(args.triplets, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    # Load base model and freeze
    base = BaseModel(embed_dim=args.embed_dim, pretrained=True)
    base.load_state_dict(torch.load(args.base_model, map_location='cpu'))
    base = base.to(device)
    base.eval()
    for param in base.parameters():
        param.requires_grad = False

    # Bi-LSTM module
    # hidden_dim=512 per paper (bidirectional -> 1024 before FC)
    bilstm = BiLSTMModule(input_dim=2048, hidden_dim=512, output_dim=args.embed_dim, num_layers=args.num_layers)
    bilstm = bilstm.to(device)

    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adagrad(bilstm.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        bilstm.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"Resume từ epoch {start_epoch}")

    bilstm.train()
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        for batch_idx, (seqs, pos_imgs, neg_imgs) in enumerate(loader):
            # seqs: (B, T, C, H, W) with T=20
            seqs = seqs.to(device)          # B x 20 x 3 x 299 x 299
            pos_imgs = pos_imgs.to(device)  # B x 3 x 299 x 299
            neg_imgs = neg_imgs.to(device)

            B, T, C, H, W = seqs.shape
            # Extract high-level features for each time step using frozen base model
            # We reshape to (B*T, C, H, W) to process all steps at once
            seq_flat = seqs.view(B * T, C, H, W)
            with torch.no_grad():
                features_flat = base.feature_extractor(seq_flat)   # (B*T, 2048)
            features = features_flat.view(B, T, -1)                # (B, T, 2048)

            # Forward through Bi-LSTM -> embeddings for all timesteps
            emb_s_all = bilstm(features)       # (B, T, D)
            emb_p = base(pos_imgs)             # (B, D)
            emb_n = base(neg_imgs)             # (B, D)

            # Vectorized triplet loss across all timesteps
            B, T, D = emb_s_all.shape
            emb_s_flat = emb_s_all.view(B * T, D)
            emb_p_flat = emb_p.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
            emb_n_flat = emb_n.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
            loss = criterion(emb_s_flat, emb_p_flat, emb_n_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

        # Save checkpoint every args.save_every epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"bilstm_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': bilstm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Lưu model cuối cùng
    torch.save(bilstm.state_dict(), args.save)
    print(f"Bi-LSTM model saved to {args.save}")


if __name__ == '__main__':
    train()