import numpy as np
from bresenham import bresenham
import scipy.ndimage
import random


def mydrawPNG(vector_images, Side=256, num_steps=20):
    """
    vector_images: list of stroke point arrays (Nx3)
    Side: kích thước ảnh vuông
    num_steps: số bước ảnh mong muốn (kể cả ảnh cuối cùng hoàn chỉnh)
    Trả về: list of ảnh (dạng numpy array 2D grayscale)
    """
    raster_images = []
    for vector_image in vector_images:
        total_points = len(vector_image)
        if total_points == 0:
            continue
        # Các chỉ mục điểm vẽ mà tại đó ta sẽ lưu ảnh (bỏ qua điểm đầu tiên)
        # Dùng np.linspace để lấy num_steps-1 điểm phân bố đều trong khoảng [0, total_points-1]
        step_indices = np.linspace(0, total_points - 1, num_steps, dtype=int)[1:]
        # Khởi tạo ảnh trắng
        raster_image = np.zeros((Side, Side), dtype=np.float32)
        initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
        current_step_idx = 0
        for i in range(total_points):
            # Nếu bắt đầu nét mới (state == 1)
            if i > 0 and vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
            # Vẽ đoạn thẳng từ (initX, initY) đến điểm hiện tại
            cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
            for cord in cordList:
                if 0 < cord[0] < Side and 0 < cord[1] < Side:
                    raster_image[cord[1], cord[0]] = 255.0
            initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
            # Nếu đến chỉ mục cần lưu ảnh
            if current_step_idx < len(step_indices) and i >= step_indices[current_step_idx]:
                # Lưu ảnh tại thời điểm này (sau khi vẽ xong điểm thứ i)
                dilated = scipy.ndimage.binary_dilation(raster_image) * 255.0
                raster_images.append(dilated)
                current_step_idx += 1
        # Đảm bảo có đủ num_steps ảnh (nếu thiếu thì thêm ảnh cuối cùng)
        while len(raster_images) < num_steps:
            raster_images.append(scipy.ndimage.binary_dilation(raster_image) * 255.0)
    return raster_images, []


def Preprocess_QuickDraw_redraw(vector_images, side = 256.0):
    vector_images = vector_images.astype(float)
    vector_images[:, :2] = vector_images[:, :2] / np.array([256, 256])
    vector_images[:,:2] = vector_images[:,:2] * side
    vector_images = np.round(vector_images)
    return vector_images

def redraw_Quick2RGB(vector_images):
    vector_images_C = Preprocess_QuickDraw_redraw(vector_images)
    raster_images, Sample_len = mydrawPNG([vector_images_C])
    return raster_images,  Sample_len
