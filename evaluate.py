"""
Evaluate Bi-LSTM model on test sequences (20 steps per episode).

Computes:
- A@1
- A@5
- A@10
- m@A
- m@B

Optimizations:
- gallery embedding cache
- batch feature extraction
- inference_mode()
- fast rank computation (no argsort)
- checkpoint compatibility
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from model import BaseModel, BiLSTMModule


CACHE_PATH = "gallery_cache.pth"


# =========================================================
# LOAD GALLERY
# =========================================================
def load_gallery(gallery_dir, model, transform, device):
    """
    Build dictionary:
        photo_id -> embedding

    Uses cache if available.
    """

    model = model.to(device)
    model.eval()

    if os.path.exists(CACHE_PATH):
        print(f"Loading gallery cache from {CACHE_PATH}")
        return torch.load(CACHE_PATH, map_location='cpu')

    print("Building gallery embeddings...")

    gallery = {}

    with torch.inference_mode():

        for img_path in Path(gallery_dir).glob("*.png"):

            img = Image.open(img_path).convert("RGB")

            img_t = transform(img).unsqueeze(0).to(device)

            emb = model(img_t).squeeze(0).cpu()

            gallery[img_path.stem] = emb

    torch.save(gallery, CACHE_PATH)

    print(f"Saved gallery cache to {CACHE_PATH}")

    return gallery


# =========================================================
# EVALUATION
# =========================================================
def evaluate_sequence(
    model_bilstm,
    base_model,
    seq_csv,
    gallery,
    transform,
    device,
    num_steps=20
):
    """
    Evaluate all sequences.

    Returns:
        dict with A@1, A@5, A@10, m@A, m@B
    """

    base_model.eval()
    model_bilstm.eval()

    gallery_ids = list(gallery.keys())

    gallery_vecs = torch.stack(
        [gallery[pid] for pid in gallery_ids]
    ).to(device)

    N = len(gallery_ids)

    # faster lookup
    id2idx = {
        pid: idx
        for idx, pid in enumerate(gallery_ids)
    }

    all_ranks = []

    with torch.inference_mode():

        with open(seq_csv, 'r', encoding='utf-8') as f:

            reader = csv.reader(f)

            for idx, row in enumerate(reader):

                if idx % 100 == 0:
                    print(f"Processed {idx} episodes...")

                if len(row) < 2:
                    continue

                seq_json, pos_path = row[0], row[1]

                step_paths = json.loads(seq_json)

                if len(step_paths) != num_steps:
                    print(
                        f"Warning: sequence has {len(step_paths)} "
                        f"steps, expected {num_steps}"
                    )
                    continue

                # =================================================
                # BATCH FEATURE EXTRACTION
                # =================================================
                imgs = []

                for p in step_paths:

                    img = Image.open(p).convert("RGB")

                    imgs.append(transform(img))

                imgs = torch.stack(imgs).to(device)

                # (T, 2048)
                features = base_model.feature_extractor(imgs)

                true_id = Path(pos_path).stem

                gt_index = id2idx.get(true_id, None)

                # =================================================
                # EVALUATE EACH STEP
                # =================================================
                for t in range(num_steps):

                    prefix = features[:t + 1].unsqueeze(0)

                    # (1, D)
                    emb = model_bilstm(prefix)[:, -1, :]

                    # (N,)
                    dist = torch.cdist(
                        emb,
                        gallery_vecs
                    ).squeeze(0)

                    if gt_index is None:

                        rank = N

                    else:
                        # faster than argsort()
                        rank = (
                            dist < dist[gt_index]
                        ).sum().item() + 1

                    all_ranks.append(rank)

    # =========================================================
    # METRICS
    # =========================================================
    total_queries = len(all_ranks)

    results = {}

    # A@k
    for k in (1, 5, 10):

        correct = sum(
            1 for r in all_ranks if r <= k
        )

        results[f'A@{k}'] = correct / total_queries

    # m@A
    percentiles = [

        (
            1 - (r - 1) / (N - 1)
        ) * 100

        if N > 1 else 100.0

        for r in all_ranks
    ]

    results['m@A'] = (
        sum(percentiles) / total_queries
    )

    # m@B (MRR)
    results['m@B'] = (
        sum(1.0 / r for r in all_ranks)
        / total_queries
    )

    return results


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        '--base-model',
        required=True,
        help='Path to base model .pth'
    )

    p.add_argument(
        '--bilstm-model',
        required=True,
        help='Path to Bi-LSTM checkpoint .pth'
    )

    p.add_argument(
        '--seq-csv',
        required=True,
        help='CSV with test sequences'
    )

    p.add_argument(
        '--gallery-dir',
        required=True,
        help='Directory with gallery images'
    )

    p.add_argument(
        '--device',
        default='cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    p.add_argument(
        '--embed-dim',
        type=int,
        default=64
    )

    return p.parse_args()


# =========================================================
# MAIN
# =========================================================
def main():

    args = parse_args()

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # =====================================================
    # LOAD BASE MODEL
    # =====================================================
    print("Loading base model...")

    base = BaseModel(
        embed_dim=args.embed_dim,
        pretrained=True
    )

    base.load_state_dict(
        torch.load(
            args.base_model,
            map_location='cpu'
        )
    )

    base = base.to(device)

    base.eval()

    # =====================================================
    # LOAD Bi-LSTM
    # =====================================================
    print("Loading Bi-LSTM model...")

    bilstm = BiLSTMModule(
        input_dim=2048,
        hidden_dim=512,
        output_dim=args.embed_dim,
        num_layers=2
    )

    checkpoint = torch.load(
        args.bilstm_model,
        map_location='cpu'
    )

    # compatible with both:
    # - checkpoint dict
    # - pure state_dict
    if 'model_state_dict' in checkpoint:

        bilstm.load_state_dict(
            checkpoint['model_state_dict']
        )

    else:

        bilstm.load_state_dict(checkpoint)

    bilstm = bilstm.to(device)

    bilstm.eval()

    # =====================================================
    # OPTIONAL: torch.compile()
    # =====================================================
    if hasattr(torch, "compile"):

        try:
            base = torch.compile(base)
            bilstm = torch.compile(bilstm)

            print("torch.compile enabled")

        except Exception:
            print("torch.compile skipped")

    # =====================================================
    # LOAD GALLERY
    # =====================================================
    print("Loading gallery...")

    gallery = load_gallery(
        args.gallery_dir,
        base,
        transform,
        device
    )

    print(f"Gallery size: {len(gallery)}")

    # =====================================================
    # EVALUATE
    # =====================================================
    print("Evaluating on test sequences...")

    results = evaluate_sequence(
        bilstm,
        base,
        args.seq_csv,
        gallery,
        transform,
        device
    )

    # =====================================================
    # RESULTS
    # =====================================================
    print("\n===== Evaluation Results =====")

    for k in (1, 5, 10):

        print(
            f"A@{k}: "
            f"{results[f'A@{k}']:.4f}"
        )

    print(f"m@A: {results['m@A']:.4f}")

    print(f"m@B: {results['m@B']:.4f}")


if __name__ == '__main__':
    main()
