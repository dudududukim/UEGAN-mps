import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

def split_image(image_path, save_dir, prefix):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    h_half, w_half = h // 2, w // 2
    
    sub_images = [
        (img[:h_half, :w_half], "_1"),
        (img[:h_half, w_half:], "_2"),
        (img[h_half:, :w_half], "_3"),
        (img[h_half:, w_half:], "_4"),
    ]
    
    filenames = []
    for sub_img, suffix in sub_images:
        filename = f"{prefix}{suffix}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, sub_img)
        filenames.append(save_path)
    
    return filenames

def process_images(src_dir, train_dir, test_dir, val_dir):
    
    images = []
    for file in os.listdir(src_dir):
        if file.lower().endswith(('png', 'jpg', 'jpeg')) and not file.startswith("._"):
            image_path = os.path.join(src_dir, file)
            prefix = os.path.splitext(file)[0]
            images.extend(split_image(image_path, train_dir, prefix))               # split_image 여기서 train에 들어있는 4등분 이미지들의 list(paths)가 넘어옴 / extend는 배열의 요소가 해당 배열로 복사됨
    
    train_images, test_images = train_test_split(images, test_size=0.1, random_state=42)
    val_images, final_test_images = train_test_split(test_images, test_size=0.8, random_state=42)
    
    for img_path in train_images:
        shutil.move(img_path, os.path.join(train_dir, os.path.basename(img_path)))
    for img_path in val_images:
        shutil.move(img_path, os.path.join(val_dir, os.path.basename(img_path)))
    for img_path in final_test_images:
        shutil.move(img_path, os.path.join(test_dir, os.path.basename(img_path)))

def main():
    base_dir = "data/d2f"
    og_dir = "data/og"
    
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    val_dir = os.path.join(base_dir, "val")
    
    categories = ["digital", "film"]
    for category in categories:
        src_path = os.path.join(og_dir, category)
        train_save_path = os.path.join(train_dir, category)
        test_save_path = os.path.join(test_dir, category)
        val_save_path = os.path.join(val_dir, category)
        
        os.makedirs(train_save_path, exist_ok=True)
        os.makedirs(test_save_path, exist_ok=True)
        os.makedirs(val_save_path, exist_ok=True)
        
        process_images(src_path, train_save_path, test_save_path, val_save_path)

if __name__ == "__main__":
    main()
