import os
import csv
import random
import re
from pathlib import Path

rendered_dir = Path("rendered_sketches")   # nơi chứa các file step
photo_dir = Path("ChairV2\\ChairV2\\photo")        # thư mục ảnh thật (điều chỉnh theo của bạn)
output_csv = "base_triplets.csv"

# Lấy tất cả các sketch_id (bỏ hậu tố step)
sketch_ids = set()
for f in rendered_dir.glob("*.png"):
    name = f.stem
    if "_step" in name:
        base = name.rsplit("_step", 1)[0]
        sketch_ids.add(base)

# Lấy danh sách ảnh thật
photo_files = list(photo_dir.glob("*.png"))
photo_ids = [p.stem for p in photo_files]

print(f"DEBUG: rendered_dir={rendered_dir} exists={rendered_dir.exists()}")
print(f"DEBUG: photo_dir={photo_dir} exists={photo_dir.exists()} files={len(photo_files)}")
print(f"DEBUG: found {len(photo_ids)} photo ids, sample: {photo_ids[:10]}")

rows = []
for sid in sketch_ids:
    # Tìm file hoàn chỉnh _step20.png (yêu cầu render phải tạo đủ 20 bước)
    complete_path = rendered_dir / f"{sid}_step20.png"
    if not complete_path.exists():
        # debug: missing exact step20 image
        # print(f"DEBUG: missing step20 for sketch_id={sid}")
        continue

    # Tìm ảnh thật tương ứng: xử lý các tiền tố/underscore khác nhau
    raw = sid.lstrip('_')
    if raw.startswith('train_') or raw.startswith('test_'):
        raw = raw.split('_', 1)[1]
    # Nếu raw có hậu tố dạng _1 (instance), bỏ nó để khớp file ảnh nếu cần
    photo_candidate = re.sub(r'_\d+$', '', raw)
    pos_path = photo_dir / f"{photo_candidate}.png"
    if not pos_path.exists():
        # thử dùng raw (không xóa hậu tố) như phương án dự phòng
        alt = photo_dir / f"{raw}.png"
        if alt.exists():
            pos_path = alt
        else:
            # debug: log missing positive mapping
            print(f"DEBUG: no positive for sketch_id={sid} -> tried {photo_candidate}.png and {raw}.png")
            continue

    # Chọn ngẫu nhiên negative (khác positive)
    neg_candidates = [p for p in photo_files if p.stem != pos_path.stem]
    if not neg_candidates:
        continue
    neg_path = random.choice(neg_candidates)
    rows.append([str(complete_path), str(pos_path), str(neg_path)])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

    print(f"Đã tạo {len(rows)} triplets trong {output_csv}")

    if len(rows) == 0:
        print("DEBUG: No triplets generated. Example rendered files (first 20):")
        sample = list(rendered_dir.glob("*.png"))[:20]
        for s in sample:
            print("  ", s.name)