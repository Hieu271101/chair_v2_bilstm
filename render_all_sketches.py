import pickle
import os
import numpy as np
from PIL import Image
from render_sketch_chairv2 import redraw_Quick2RGB   # dùng cho ChairV2
# Đường dẫn đến file stroke data (đã có sẵn trong repo)
coordinate_path = "ChairV2_Coordinate"   # hoặc "ShoeV2_Coordinate"
output_dir = "rendered_sketches"  # nơi lưu ảnh PNG
os.makedirs(output_dir, exist_ok=True)

# Đọc dữ liệu stroke
with open(coordinate_path, 'rb') as f:
    data = pickle.load(f)

# Duyệt từng sketch episode
for sketch_id, strokes in data.items():
    print(f"Rendering {sketch_id} ...")
    # Render 20 bước ảnh (dạng numpy array RGB)
    images, _ = redraw_Quick2RGB(strokes)   # list of 20 HxWx3
    # Lưu từng bước thành file PNG
    for step, img_array in enumerate(images, start=1):
        arr = np.asarray(img_array)
        arr = np.clip(arr, 0, 255)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            img = Image.fromarray(arr, mode='L')
        else:
            img = Image.fromarray(arr)
        # Tạo tên file: ví dụ train_00140128_v1_step1.png
        safe_id = sketch_id.replace('/', '_')   # train/00140128_v1 -> train_00140128_v1
        img.save(f"{output_dir}/{safe_id}_step{step}.png")
    # Chỉ test 10 episode đầu (nếu muốn dừng sớm)
    # if list(data.keys()).index(sketch_id) > 10: break