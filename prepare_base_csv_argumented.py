import csv
import random
from pathlib import Path

# Đường dẫn thư mục ảnh thật (gallery)
photo_dir = Path("ChairV2/ChairV2/photo")
all_photos = [p.stem for p in photo_dir.glob("*.png")]
print(f"Total photos: {len(all_photos)}")

# Đọc các cặp (sketch_path, positive_id) từ base_triplets.csv hiện tại
# (có thể có nhiều sketch khác nhau cho cùng một positive)
pairs = set()
with open("base_triplets.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            sketch_path = row[0]
            pos_id = Path(row[1]).stem
            pairs.add((sketch_path, pos_id))

print(f"Unique (sketch, positive) pairs: {len(pairs)}")

# Số negative cần tạo cho mỗi cặp
NEG_PER_PAIR = 20  # có thể tăng lên 30-50 nếu muốn

new_triplets = []
for sketch_path, pos_id in pairs:
    # Danh sách negative candidates (tất cả ảnh khác pos_id)
    neg_candidates = [pid for pid in all_photos if pid != pos_id]
    # Nếu số negative ít hơn NEG_PER_PAIR thì lấy hết
    num_neg = min(NEG_PER_PAIR, len(neg_candidates))
    selected_negs = random.sample(neg_candidates, num_neg)
    for neg_id in selected_negs:
        pos_path = str(photo_dir / f"{pos_id}.png")
        neg_path = str(photo_dir / f"{neg_id}.png")
        new_triplets.append([sketch_path, pos_path, neg_path])

# Ghi ra file mới
output_csv = "base_triplets_augmented.csv"
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(new_triplets)

print(f"Generated {len(new_triplets)} triplets (was {len(pairs)} pairs)")
print(f"Saved to {output_csv}")