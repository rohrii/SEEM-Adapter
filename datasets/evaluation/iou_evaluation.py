from collections import OrderedDict
import copy
import itertools
import logging
import os
from typing import Optional
import contextlib
import io
from pycocotools.coco import COCO
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm
import torch
from pycocotools import mask as coco_mask


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
    return masks.to(torch.device('cuda'))


class IouEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._logger = logging.getLogger(__name__)
        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._distributed = True
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

        self._predictions = []
            
    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {}

            if "instances" in output:
                prediction["pred_masks"] = output["instances"].get("pred_masks")
        
            prediction["image_id"] = input["image_id"]
            prediction["height"] = input["height"]
            prediction["width"] = input["width"]
            
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
            print("evaluating...")
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = {"mIoU": 0}
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
    
    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the tasks.
        """
        ious = []
        for prediction in predictions:
            image_shape = (prediction["height"], prediction["width"])
            annos = self._coco_api.loadAnns(self._coco_api.getAnnIds(imgIds=[prediction["image_id"]]))
            gt_masks = convert_coco_poly_to_mask([obj["segmentation"] for obj in annos], *image_shape)
            ious.append(self._compute_iou(prediction["pred_masks"], gt_masks))
        self._results["mIoU"] = torch.stack(ious).mean().to("cpu").item()

    def _compute_iou(self, pred_masks, gt_masks):
        """
        Args:
            instances: Instances
            gt_masks: Tensor of shape (N, H, W), where N is the number of objects.
                H, W are the height and width of the predicted mask.
        Returns:
            Tensor of shape (N,), where N is the number of instances.
        """
        if len(gt_masks) == 0:
            return torch.zeros(len(pred_masks))
        gt_masks = gt_masks[:, None, :, :]
        gt_masks = torch.as_tensor(gt_masks, dtype=torch.bool, device=gt_masks.device)
        pred_masks = torch.as_tensor(pred_masks, dtype=torch.bool, device=pred_masks.device)
        # flatten all masks
        gt_masks = gt_masks.reshape(-1, gt_masks.shape[-2], gt_masks.shape[-1])
        pred_masks = pred_masks.reshape(-1, pred_masks.shape[-2], pred_masks.shape[-1])
        # flatten all masks
        gt_masks = gt_masks.reshape(-1, gt_masks.shape[-2], gt_masks.shape[-1])
        pred_masks = pred_masks.reshape(-1, pred_masks.shape[-2], pred_masks.shape[-1])
        # intersect = (gt_masks & pred_masks).sum(dim=(1, 2)).float()
        # union = (gt_masks | pred_masks).sum(dim=(1, 2)).float()
        # iou = intersect / union
        iou = (gt_masks & pred_masks).sum(dim=(1, 2)).float() / (gt_masks | pred_masks).sum(dim=(1, 2)).float()
        print(iou)
        return iou