import os
from detectron2.data.datasets import register_coco_instances

DATA_SET_NAME = "chestx_instance"
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
DATA_SET_ROOT = os.path.join(os.path.expanduser("~"), "masters-thesis", "datasets", "chestx-det")

# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "train", "images")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "train" + ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# TEST SET
TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "test", "images")
TEST_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "test" + ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TEST_DATA_SET_NAME,
    metadata={},
    json_file=TEST_DATA_SET_ANN_FILE_PATH,
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)