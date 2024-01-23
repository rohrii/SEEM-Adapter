# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch
import cv2

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask

from modeling.utils import configurable

__all__ = ["TrashcanDatasetMapper"]

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def get_colored_rows_info(rgb_image):
    """
    Obtain the index of the first and last colored row as well as the number of colored rows.

    Parameters:
    image (numpy.ndarray): The input image as a numpy array.

    Returns:
    int: Index of the first colored row.
    int: Total number of colored rows.
    """
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # For grayscale images, a row is considered black if its sum is < 1000
    non_black_rows = gray_image.sum(axis=1) > 1000

    # Get the indices of the non-black (colored) rows
    colored_row_indices = np.where(non_black_rows)[0]

    # Get the index of the first colored row
    first_colored_row_index = colored_row_indices[0]

    # Get the total number of colored rows
    num_colored_rows = len(colored_row_indices)

    return first_colored_row_index, num_colored_rows

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []

    if cfg_input['RANDOM_FLIP'] != "none" and is_train:
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    if is_train:
        augmentation.append(T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ))

    return augmentation


# This is specifically designed for the COCO dataset.
class TrashcanDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by SEEM.
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            tfm_gens,
            image_format,
            mask_format
    ):
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[Trashcan DatasetMapper] Full TransformGens used: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.mask_format = mask_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "mask_format": cfg['INPUT']['MASK_FORMAT']
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            Contains binary mask instead of segmentation polygon or RLE

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # calculate the black parts of the image (Trashcan dataset only)
        padding_top, colored_img_height = get_colored_rows_info(image)

        # Crop the image to only contain the roi
        tfm_gens = [T.CropTransform(x0=0, y0=padding_top, w=image.shape[1], h=colored_img_height)]
        tfm_gens.extend(self.tfm_gens)

        padding_mask = np.ones(image.shape[:2], dtype="uint8")

        image, transforms = T.apply_transform_gens(tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if not "annotations" in dataset_dict: return dataset_dict
        
        annos = []
        for anno in dataset_dict.pop("annotations"):
            if anno.get("iscrowd", 0) == 1: continue
            new_anno = utils.transform_instance_annotations(anno, transforms, image_shape)
            annos.append(new_anno)        
        
        instances = utils.annotations_to_instances(annos, image_shape, self.mask_format)
        
        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        # Need to filter empty instances first (due to augmentation)
        instances = utils.filter_empty_instances(instances)

        # Generate masks from polygon
        h, w = instances.image_size
        # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
        if hasattr(instances, 'gt_masks'):
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks

        dataset_dict["instances"] = instances

        return dataset_dict
