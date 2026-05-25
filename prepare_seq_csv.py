import json
import csv
import random
import re
from pathlib import Path

rendered_dir = Path("rendered_sketches")
photo_dir = Path("ChairV2/ChairV2/photo")
output_csv = "seq_triplets.csv"

# Get all sketch_ids (remove the step suffix)
sketch_ids = set()
for f in rendered_dir.glob("*.png"):
    name = f.stem
    if "_step" in name:
        base = name.rsplit("_step", 1)[0]
        sketch_ids.add(base)

photo_files = list(photo_dir.glob("*.png"))

def step_num(p):
    m = re.search(r"_step(\d+)$", p.stem)
    return int(m.group(1)) if m else 0

rows = []
for sid in sketch_ids:

    # Get all step files for this sketch
    step_files = list(rendered_dir.glob(f"{sid}_step*.png"))
    if not step_files:
        continue

    step_files_sorted = sorted(step_files, key=step_num)

    # Check if there are 20 steps (after editing, the render must have 20 steps)
    if len(step_files_sorted) != 20:
        print(f"Warning: {sid} only has {len(step_files_sorted)} steps, skip")
        continue

    step_paths = [str(p) for p in step_files_sorted]  # Get all 20 steps

    # Get the corresponding real image name
    raw = sid.lstrip('_')
    if raw.startswith('train_') or raw.startswith('test_'):
        photo_candidate = raw.split('_', 1)[1]
    else:
        photo_candidate = raw

    # Remove the suffix _\d+ (if any) to match the actual image file name
    photo_candidate = re.sub(r'_\d+$', '', photo_candidate)

    pos_path = photo_dir / f"{photo_candidate}.png"
    if not pos_path.exists():
        # Try with the original name (no processing)
        pos_path = photo_dir / f"{raw}.png"
        if not pos_path.exists():
            print(f"Missing positive for {sid}: {photo_candidate}.png not found")
            continue

    # Select negative other than positive
    neg_candidates = [p for p in photo_files if p.stem != pos_path.stem]
    if not neg_candidates:
        continue
    neg_path = random.choice(neg_candidates)
    seq_json = json.dumps(step_paths)
    rows.append([seq_json, str(pos_path), str(neg_path)])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Created {len(rows)} sequence triplet in {output_csv}")