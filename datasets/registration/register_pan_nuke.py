import os
from detectron2.data.datasets import register_coco_instances

DATA_SET_NAME = "pan_nuke"
DATA_SET_ROOT = os.path.join(os.path.expanduser("~"), "masters-thesis", "datasets", "pan-nuke", "dataset")

# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "train.json")

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# VAL SET
VAL_DATA_SET_NAME = f"{DATA_SET_NAME}-val"
VAL_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "val")
VAL_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "val.json")

register_coco_instances(
    name=VAL_DATA_SET_NAME,
    metadata={},
    json_file=VAL_DATA_SET_ANN_FILE_PATH,
    image_root=VAL_DATA_SET_IMAGES_DIR_PATH
)