import torch
import csv
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model import BaseModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transform giống hệt khi train base model
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load base model (thay đổi đường dẫn nếu cần)
    base = BaseModel(embed_dim=64, pretrained=True)
    base.load_state_dict(torch.load('chair_base_augmented.pth', map_location='cpu'))
    base = base.to(device)
    base.eval()

    # Load gallery
    gallery_dir = "ChairV2/ChairV2/photo"
    gallery = {}
    for img_path in Path(gallery_dir).glob("*.png"):
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = base(img_t).squeeze(0).cpu()
        gallery[img_path.stem] = emb

    gallery_ids = list(gallery.keys())
    gallery_vecs = torch.stack([gallery[pid] for pid in gallery_ids]).to(device)
    print(f"Gallery size: {len(gallery)}")

    # Đọc file CSV test và lấy sketch step20
    ranks = []
    with open("seq_triplets_fixed.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            seq_json = row[0]
            pos_path = row[1]
            step_paths = json.loads(seq_json)
            if len(step_paths) != 20:
                continue
            last_sketch_path = step_paths[-1]  # lấy bước thứ 20
            true_id = Path(pos_path).stem
            if true_id not in gallery:
                continue
            # Embed sketch
            img = Image.open(last_sketch_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb_query = base(img_t)
            # Khoảng cách
            dist = torch.cdist(emb_query, gallery_vecs).squeeze(0)
            gt_idx = gallery_ids.index(true_id)
            rank = (dist < dist[gt_idx]).sum().item() + 1
            ranks.append(rank)

    # Tính metrics
    total = len(ranks)
    print(f"Total queries: {total}")
    for k in (1, 5, 10):
        acc = sum(1 for r in ranks if r <= k) / total
        print(f"A@{k}: {acc:.4f}")

    mrr = sum(1.0 / r for r in ranks) / total
    print(f"m@B (MRR): {mrr:.4f}")

if __name__ == "__main__":
    main()