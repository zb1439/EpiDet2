import itertools
import os
import json

import torch
import detectron2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random
from fvcore.common.file_io import PathManager

from .coco_evaluation import COCOEvaluator, instances_to_coco_json, _evaluate_predictions_on_coco
from detectron2.structures import Instances, Boxes
from detectron2.data.datasets.builtin_meta import _get_coco_fewshot_instances_meta, _get_coco_instances_meta

class COCOPsuedoEpisodicEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed,
                 output_dir=None, eval_classes='base', way=3):
        super(COCOPsuedoEpisodicEvaluator, self).__init__(dataset_name, cfg, distributed, output_dir)
        self.way = way
        metadata = _get_coco_fewshot_instances_meta()
        if eval_classes == 'base':
            self._eval_classes = metadata["base_ids"]
            self._cls_map = metadata["base_dataset_id_to_contiguous_id"]
            self._eval_classes_names = metadata["base_classes"]
            self._eval_classes_contiguous = [self._cls_map[id] for id in self._eval_classes]
            self._cls_map_reverse = {v: k for k, v in self._cls_map.items()}
        elif eval_classes == 'novel':
            self._eval_classes = metadata["novel_ids"]
            self._cls_map = metadata["novel_dataset_id_to_contiguous_id"]
            self._eval_classes_names = metadata["novel_classes"]
            self._eval_classes_contiguous = [self._cls_map[id] for id in self._eval_classes]
            self._cls_map_reverse = {v: k for k, v in self._cls_map.item()}
        else:
            metadata = _get_coco_instances_meta()
            self._eval_classes = metadata["thing_ids"]
            self._cls_map = metadata["thing_dataset_id_to_contiguous_id"]
            self._eval_classes_names = metadata["thing_classes"]
            self._eval_classes_contiguous = [self._cls_map[id] for id in self._eval_classes]
            self._cls_map_reverse = {v: k for k, v in self._cls_map.item()}

    def process(self, inputs, outputs):
        """ This function samples repeatedly to simulate episodic evaluation.
        """
        for input, output in zip(inputs, outputs):
            assert "instances" in output
            instances = output["instances"].to(self._cpu_device)
            rest_classes = torch.unique(input["instances"].gt_classes.to(self._cpu_device)).numpy().tolist()
            while len(rest_classes) > 0:
                category = random.choice(rest_classes)
                epi_classes = [category]
                for _ in range(self.way - 1):
                    while True:
                        other_cat = random.choice(self._eval_classes_contiguous)
                        if other_cat not in epi_classes:
                            epi_classes.append(other_cat)
                            break

                prediction = {"image_id": input["image_id"]}
                epi_instances = Instances(instances.image_size)
                pickups = torch.zeros_like(instances.pred_classes, dtype=torch.bool)
                for cls in epi_classes:
                    pickups += instances.pred_classes == cls
                if torch.nonzero(pickups).numel() > 0:
                    print("warning: no gt class pred found during this episode")
                    for cat in epi_classes:
                        if cat in rest_classes:
                            rest_classes.remove(cat)

                epi_instances.pred_classes = instances.pred_classes[pickups]
                epi_instances.pred_boxes = instances.pred_boxes[pickups]
                epi_instances.scores = instances.scores[pickups]
                if hasattr(instances, "pred_masks"):
                    epi_instances.pred_masks = instances.pred_masks[pickups]

                prediction["instances"] = instances_to_coco_json(epi_instances, input["image_id"])
                self._predictions.append(prediction)

                for cat in epi_classes:
                    if cat in rest_classes:
                        rest_classes.remove(cat)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        # General detection evaluation
        if not self._is_splits:
            for task in sorted(tasks):
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas,
                        catIds=self._eval_classes
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results(
                    coco_eval, task, class_names=self._metadata.get("thing_classes")
                )
                self._results[task] = res
        # Few-Shot detection evaluation
        else:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", catIds=classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)
            # if evaluate on novel dataset only
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]

VOC_THING_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse',
                 'sheep', 'airplane', 'bicycle', 'boat', 'bus',
                 'car', 'motorcycle', 'train', 'bottle', 'chair',
                 'dining table', 'potted plant', 'couch', 'tv/monitor']

class VOCPsuedoEpisodicEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed,
                 output_dir=None, eval_classes='base', way=3):
        super(VOCPsuedoEpisodicEvaluator, self).__init__(dataset_name, cfg, distributed, output_dir)
        self.way = way
        metadata = _get_coco_fewshot_instances_meta()

        self._eval_classes = metadata["novel_ids"]
        self._cls_map = metadata["novel_dataset_id_to_contiguous_id"]
        self._eval_classes_names = metadata["novel_classes"]

        # This is 0-started, [0, ..., 19]
        self._eval_classes_contiguous = [self._cls_map[id] for id in self._eval_classes]
        self._cls_map_reverse = {v: k for k, v in self._cls_map.items()}

        # This is used for dataset mapping in _eval_prediction, so the keys are 0-started and values are 1-started.
        self.coco2voc = {
            0: 1, 1: 9, 2: 12, 3: 13, 4: 8, 5: 11, 6: 14, 39: 15, 10: 10, 14: 2, 15: 3,
            16: 5, 17: 6, 18: 7, 19: 4, 56: 16, 57: 19, 58: 18, 60: 17, 62: 20
        }
        # This is used for input-output mapping, both keys and values are 0-started.
        self.voc2coco = {v - 1: k for k, v in self.coco2voc.items()}
        # This is 0-started continuous coco novel class id
        self._eval_classes_contiguous_coco = [self.voc2coco[cls] for cls in self._eval_classes_contiguous]

    def process(self, inputs, outputs):
        """ This function samples repeatedly to simulate episodic evaluation.
        """
        for input, output in zip(inputs, outputs):
            assert "instances" in output
            instances = output["instances"].to(self._cpu_device)
            rest_classes = torch.unique(input["instances"].gt_classes.to(self._cpu_device)).numpy().tolist()
            rest_classes = [self.voc2coco[c] for c in rest_classes]
            while len(rest_classes) > 0:
                category = random.choice(rest_classes)
                epi_classes = [category]
                for _ in range(self.way - 1):
                    while True:
                        other_cat = random.choice(self._eval_classes_contiguous_coco)
                        if other_cat not in epi_classes:
                            epi_classes.append(other_cat)
                            break

                prediction = {"image_id": input["image_id"]}
                epi_instances = Instances(instances.image_size)

                pickups = torch.zeros_like(instances.pred_classes, dtype=torch.bool)
                # main_print(instances.pred_classes, epi_classes, rest_classes)
                for cls in epi_classes:
                    pickups += instances.pred_classes == cls
                if torch.nonzero(pickups).numel() > 0:
                    for cat in epi_classes:
                        if cat in rest_classes:
                            rest_classes.remove(cat)

                epi_instances.pred_classes = instances.pred_classes[pickups]
                epi_instances.pred_boxes = instances.pred_boxes[pickups]
                epi_instances.scores = instances.scores[pickups]
                if hasattr(instances, "pred_masks"):
                    epi_instances.pred_masks = instances.pred_masks[pickups]

                prediction["instances"] = instances_to_coco_json(epi_instances, input["image_id"])
                self._predictions.append(prediction)

                for cat in epi_classes:
                    if cat in rest_classes:
                        rest_classes.remove(cat)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # Change: unmap the category ids for VOC
        reverse_id_mapping = self.coco2voc
        for result in self._coco_results:
            category_id = result["category_id"]
            assert (
                category_id in reverse_id_mapping
            ), "A prediction has category_id={}, which is not available in the dataset.".format(
                category_id
            )
            result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        # General detection evaluation
        if not self._is_splits:
            for task in sorted(tasks):
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas,
                        catIds=self._eval_classes
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results(
                    coco_eval, task, class_names=VOC_THING_CLASSES
                )
                self._results[task] = res
        # Few-Shot detection evaluation
        else:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", catIds=classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=VOC_THING_CLASSES,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)
            # if evaluate on novel dataset only
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]



