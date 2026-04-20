import os
import cv2
import splitfolders  # pip install split-folders
import stat

# Base directory = one level up from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "split")
# Image settings
IMG_SIZE = (224, 224)  # Resize images to 224x224 (fits MobileNet, VGG, ResNet)
SPLIT_RATIO = (0.7, 0.15, 0.15)  # Train/Val/Test ratio


def preprocess_images():
    """Resize images and save them to processed folder."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for category in os.listdir(RAW_DIR):
        category_path = os.path.join(RAW_DIR, category)
        save_path = os.path.join(PROCESSED_DIR, category)
        os.makedirs(save_path, exist_ok=True)

        if not os.path.isdir(category_path):
            continue

        print(f"[INFO] Processing category: {category}")

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            try:
                img = cv2.imread(img_path)

                if img is None:
                    print(f"[WARNING] Skipping invalid file: {img_path}")
                    continue

                # Resize
                img = cv2.resize(img, IMG_SIZE)

                # Save processed image
                cv2.imwrite(os.path.join(save_path, img_name), img)

            except Exception as e:
                print(f"[ERROR] Could not process {img_path}: {e}")


import stat

def handle_remove_readonly(func, path, exc_info):
    """Force remove read-only files on Windows (OneDrive/file locks)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def split_dataset():
    """Split dataset into train/val/test folders."""
    if os.path.exists(SPLIT_DIR):
        print("[INFO] Removing old split directory...")
        import shutil
        shutil.rmtree(SPLIT_DIR, onerror=handle_remove_readonly)

    print("[INFO] Splitting dataset...")
    splitfolders.ratio(PROCESSED_DIR, output=SPLIT_DIR, seed=42, ratio=SPLIT_RATIO)
    print("[INFO] Dataset split completed.")



if __name__ == "__main__":
    print("[STEP 1] Preprocessing images...")
    preprocess_images()

    print("[STEP 2] Splitting dataset...")
    split_dataset()

    print("[DONE] Dataset is ready in 'data/split/' 🚀")
