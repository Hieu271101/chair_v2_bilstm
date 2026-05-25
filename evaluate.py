"""
Evaluate Bi-LSTM model on test sequences (20 steps per episode).
Computes A@1, A@5, A@10, m@A, m@B as in the paper.
"""
import argparse
import csv
import json
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import BaseModel, BiLSTMModule


def load_gallery(gallery_dir, model, transform, device):
    """Build dictionary of photo embeddings (all photos in gallery_dir)."""
    model = model.to(device)
    model.eval()
    gallery = {}
    with torch.no_grad():
        for img_path in Path(gallery_dir).glob("*.png"):
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            emb = model(img_t).cpu().squeeze(0)
            gallery[img_path.stem] = emb
    return gallery


def evaluate_sequence(model_bilstm, base_model, seq_csv, gallery, transform, device, num_steps=20):
    """
    Returns dict with A@1, A@5, A@10, m@A, m@B.
    For each episode, we evaluate at every step (1..20) and average metrics.
    """
    base_model.eval()
    model_bilstm.eval()
    gallery_ids = list(gallery.keys())
    gallery_vecs = torch.stack([gallery[pid] for pid in gallery_ids]).to(device)
    N = len(gallery_ids)

    # Storage for all queries (every step of every episode)
    all_ranks = []          # list of ranks (1-based) for each query

    with torch.no_grad():
        with open(seq_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx % 100 == 0:
                    print(f"Processed {idx} episodes...")
                if len(row) < 2:
                    continue
                seq_json, pos_path = row[0], row[1]
                step_paths = json.loads(seq_json)
                # Expect exactly num_steps paths (e.g., 20)
                if len(step_paths) != num_steps:
                    print(f"Warning: sequence has {len(step_paths)} steps, expected {num_steps}")
                    continue
                # Precompute features for all steps (frozen base model) on device
                features = []
                for p in step_paths:
                    img = Image.open(p).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    feat = base_model.feature_extractor(img_t)   # (1,2048)
                    features.append(feat)
                features = torch.cat(features, dim=0)   # (T, 2048)

                # For each step t (1..T), compute embedding and rank
                for t in range(num_steps):
                    prefix = features[:t+1].unsqueeze(0)   # (1, t+1, 2048)

                    # FIX: take only the last-time-step embedding from Bi-LSTM output
                    emb = model_bilstm(prefix)[:, -1, :].to(device)   # (1, D)

                    # Compute rank (keep computation on device for speed)
                    dist = torch.cdist(emb, gallery_vecs).squeeze(0)  # (N,)
                    sorted_idx = dist.argsort()
                    true_id = Path(pos_path).stem
                    try:
                        gt_index = gallery_ids.index(true_id)
                        rank = (sorted_idx == gt_index).nonzero(as_tuple=True)[0].item() + 1
                    except ValueError:
                        rank = N   # not found, worst rank

                    all_ranks.append(rank)

    # Compute metrics
    # A@k
    total_queries = len(all_ranks)
    results = {}
    for k in (1,5,10):
        correct = sum(1 for r in all_ranks if r <= k)
        results[f'A@{k}'] = correct / total_queries

    # m@A: mean ranking percentile = mean( (1 - (rank-1)/(N-1)) * 100 )
    percentiles = [(1 - (r - 1) / (N - 1)) * 100 if N > 1 else 100.0 for r in all_ranks]
    results['m@A'] = sum(percentiles) / total_queries

    # m@B: mean reciprocal rank
    results['m@B'] = sum(1.0 / r for r in all_ranks) / total_queries

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-model', required=True, help='Path to base model .pth')
    p.add_argument('--bilstm-model', required=True, help='Path to Bi-LSTM .pth')
    p.add_argument('--seq-csv', required=True, help='CSV with test sequences (same format as train)')
    p.add_argument('--gallery-dir', required=True, help='Directory with all test photos (e.g., ChairV2/testB)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--embed-dim', type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # Load base model
    base = BaseModel(embed_dim=args.embed_dim, pretrained=True)
    base.load_state_dict(torch.load(args.base_model, map_location='cpu'))
    base = base.to(device)
    base.eval()

    # Load Bi-LSTM
    bilstm = BiLSTMModule(input_dim=2048, hidden_dim=512, output_dim=args.embed_dim, num_layers=2)
    bilstm.load_state_dict(torch.load(args.bilstm_model, map_location='cpu'))
    bilstm = bilstm.to(device)
    bilstm.eval()

    # Build gallery
    print("Loading gallery...")
    gallery = load_gallery(args.gallery_dir, base, transform, device)
    print(f"Gallery size: {len(gallery)}")

    # Evaluate
    print("Evaluating on test sequences...")
    results = evaluate_sequence(bilstm, base, args.seq_csv, gallery, transform, device)
    print("\n===== Evaluation Results =====")
    for k in (1,5,10):
        print(f"A@{k}: {results[f'A@{k}']:.4f}")
    print(f"m@A: {results['m@A']:.4f}")
    print(f"m@B: {results['m@B']:.4f}")


if __name__ == '__main__':
    main()