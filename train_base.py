import argparse
import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import BaseModel, TripletLoss


class TripletImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.rows = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    self.rows.append((row[0], row[1], row[2]))
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sketch_path, pos_path, neg_path = self.rows[idx]
        sketch = Image.open(sketch_path).convert('RGB')
        pos = Image.open(pos_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')
        return self.transform(sketch), self.transform(pos), self.transform(neg)


def parse_args():
    parser = argparse.ArgumentParser(description="Train base model for FG-SBIR")
    parser.add_argument('--triplets', required=True, help='CSV with sketch,pos,neg paths')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--save', default='base_model.pth')
    parser.add_argument('--pretrained', action='store_true', default=True)
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = TripletImageDataset(args.triplets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = BaseModel(embed_dim=args.embed_dim, pretrained=args.pretrained).to(device)
    criterion = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (sketch, pos, neg) in enumerate(loader):
            sketch = sketch.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            emb_s = model(sketch)
            emb_p = model(pos)
            emb_n = model(neg)

            loss = criterion(emb_s, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx} | Loss {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"{Path(args.save).stem}_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), args.save)
    print(f"Model saved to {args.save}")


if __name__ == '__main__':
    train()
"""
Train base embedding model (CNN + attention) using triplet loss.
Triplet CSV format: sketch_path, pos_photo_path, neg_photo_path
Sketch path should point to the COMPLETE sketch (step20.png).

Example:
    python train_base.py --triplets base.csv --epochs 40 --batch-size 20 --save chair_base.pth
"""
import argparse
import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import BaseModel, TripletLoss


class TripletImageDataset(Dataset):
    """Reads triplets from CSV, each row: sketch, positive, negative."""
    def __init__(self, csv_path, transform=None):
        self.rows = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    self.rows.append((row[0], row[1], row[2]))
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sketch_path, pos_path, neg_path = self.rows[idx]
        sketch = Image.open(sketch_path).convert('RGB')
        pos = Image.open(pos_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')
        return self.transform(sketch), self.transform(pos), self.transform(neg)


def parse_args():
    parser = argparse.ArgumentParser(description="Train base model for FG-SBIR")
    parser.add_argument('--triplets', required=True, help='CSV with sketch,pos,neg paths')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs (paper uses 40)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Batch size (paper: Chair=20, Shoe=120, default 20)')
    parser.add_argument('--lr', type=float, default=0.01, help='Adagrad learning rate')
    parser.add_argument('--embed-dim', type=int, default=64, help='Output embedding dimension')
    parser.add_argument('--save', default='base_model.pth', help='Path to save model weights')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Do not use ImageNet pretrained weights')
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = TripletImageDataset(args.triplets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model = BaseModel(embed_dim=args.embed_dim, pretrained=args.pretrained).to(device)
    criterion = TripletLoss(margin=0.3)   # margin = 0.3 as in paper
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (sketch, pos, neg) in enumerate(loader):
            sketch = sketch.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            emb_s = model(sketch)
            emb_p = model(pos)
            emb_n = model(neg)

            loss = criterion(emb_s, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx} | Loss {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

        # Optional: save checkpoint every few epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"{Path(args.save).stem}_epoch{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), args.save)
    print(f"Model saved to {args.save}")


if __name__ == '__main__':
    train()