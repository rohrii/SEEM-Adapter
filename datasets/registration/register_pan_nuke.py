from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import numpy as np
from PIL import Image
import cv2

from utils.constants import PAN_NUKE_CLASSES

DATA_SET_NAME = "pan_nuke"
DATA_SET_ROOT = os.path.join(os.path.expanduser("~"), "masters-thesis", "datasets", "pan-nuke", "dataset")

# # TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "train.json")

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# # VAL SET
VAL_DATA_SET_NAME = f"{DATA_SET_NAME}-val"
VAL_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "val")
VAL_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "val.json")

register_coco_instances(
    name=VAL_DATA_SET_NAME,
    metadata={},
    json_file=VAL_DATA_SET_ANN_FILE_PATH,
    image_root=VAL_DATA_SET_IMAGES_DIR_PATH
)

# class PanNukeDataset:
#     def __init__(self, data_dir, split):
#         self.split = split
#         self.image_dir = os.path.join(data_dir, split, 'images')
#         self.mask_dir = os.path.join(data_dir, split, 'masks')
#         self.image_paths = sorted(os.listdir(self.image_dir))
#         self.mask_paths = sorted(os.listdir(self.mask_dir))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_paths[idx])
#         mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

#         image = Image.open(image_path).convert("RGB")
#         mask = np.load(mask_path)

#         annos = []
#         for class_id in range(mask.shape[-1] - 1):
#             class_mask = mask[..., class_id]
#             instance_numbers = np.unique(class_mask)
#             instance_nums = instance_numbers[1:] # remove background color

#             for inst_num in instance_nums:
#                 instance_mask = np.array(class_mask == inst_num, dtype=np.uint8)
#                 non_zero_coords = cv2.findNonZero(instance_mask)
#                 x, y, w, h = cv2.boundingRect(non_zero_coords)

#                 annos.append({
#                     "category_id": class_id,
#                     "binary_mask": instance_mask,
#                     "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
#                     "bbox_mode": 0,
#                     "is_crowd": 0
#                 })

#         return {
#             "file_name": image_path,
#             "height": image.height,
#             "width": image.width,
#             "image_id": idx,
#             "annotations": annos
#         }

# def register_custom_dataset(name, split):
#     DatasetCatalog.register(name, lambda: PanNukeDataset(DATA_SET_ROOT, split))
#     MetadataCatalog.get(name).set(thing_classes=PAN_NUKE_CLASSES, evaluator_type="coco_instance")

# register_custom_dataset(TRAIN_DATA_SET_NAME, "train")
# register_custom_dataset(VAL_DATA_SET_NAME, "val")
