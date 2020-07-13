import logging
import numpy as np
import torch
import copy

from detectron2.structures import Boxes, BoxMode, BitMasks
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, DatasetFromList, MapDataset
from detectron2.data import detection_utils as utils
from detectron2.data.build import get_detection_dataset_dicts
import detectron2.utils.comm as comm
from detectron2.utils.comm import is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data.build import InferenceSampler, RepeatFactorTrainingSampler, \
    TrainingSampler, build_batch_data_loader, trivial_batch_collator

def main_print(*msg):
    if comm.is_main_process():
        print(*msg)

def build_support_transform_gen(cfg, is_train):
    """ See det2/data/detection_utils.py build_transform_gen()
    """
    min_size = cfg.METARCNN.SUPPORT.MIN_SIZE
    max_size = cfg.METARCNN.SUPPORT.MAX_SIZE
    sample_style = "choice"

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens for support used in training: " + str(tfm_gens))
    return tfm_gens

class SupportDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.FORMAT
        self.concat_mask = cfg.METARCNN.CONCAT_MASK
        self.mask_format = "bitmask"
        self.tfm_gens = build_support_transform_gen(cfg, is_train)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.mask_format
        )
        if len(instances.gt_masks) == 0:
            print(instances)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        assert type(dataset_dict["instances"].gt_masks) == BitMasks and \
            len(dataset_dict["instances"].gt_masks) == 1, \
            "BitMask should only contain 1 mask for support json, but {} were found (original annotation: {})".format(
                dataset_dict["instances"], annos)
        if self.concat_mask:
            mask = dataset_dict["instances"].gt_masks.tensor[0]

            assert mask.size(0) == dataset_dict["image"].size(1) and \
                mask.size(1) == dataset_dict["image"].size(2)
            dataset_dict["annotation"] = mask.float()
        else:
            mask = torch.zeros(image_shape, dtype="float").to(image.device)
            x1, y1, x2, y2 = dataset_dict["instances"].gt_boxes.tensor[0]
            mask[y1:y2, x1:x2] = 1.0
            dataset_dict["annotation"] = mask
        dataset_dict["class"] = dataset_dict["instances"].gt_classes[0].item()
        dataset_dict.pop("instances")

        return dataset_dict

class SupportDataset(MapDataset):
    """
    This class extends DatasetMapper to group the entries of each class as the original implementation.
    https://github.com/yanxp/MetaR-CNN/blob/0e54c48505a0fd472eec3885d1ea2a80852cf681/lib/datasets/metadata.py
    Add parallel feature to speed up grouping process.
    """
    def __init__(self, dataset, map_func, classes, length=100000):
        super(SupportDataset, self).__init__(dataset, map_func)
        self.length = length
        self.real_length = 0
        self.classes = classes
        self.classwise_dataset = {}
        for cls in self.classes:
            self.classwise_dataset[cls] = []
        self.group()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert self.real_length > 0
        idx = idx % self.real_length
        item = []
        for k in self.classes:
            item.append(self.classwise_dataset[k][idx])
        return item

    def group(self):
        # TODO: try to this with data parallelism 'cause the map func is time consuming.
        import multiprocessing as mp
        from joblib import Parallel, delayed
        from tqdm import tqdm
        cpu_count = mp.cpu_count() // 4

        dataset = [copy.deepcopy(self._dataset[i]) for i in range(len(self._dataset))]
        del self._dataset
        if comm.is_main_process():
            dataset = tqdm(dataset)
        after_map = Parallel(cpu_count)(delayed(self._map_func)(data) for data in dataset)

        for data in after_map:
            self.classwise_dataset[data["class"]].append(data)
        self.real_length = min([len(v) for v in self.classwise_dataset.values()])

    def get_real_length(self):
        return self.real_length

    def get_classes(self):
        return sorted(list(self.classwise_dataset.keys()))


def build_metarcnn_train_loader(cfg, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    if cfg.METARCNN.PHASE == 1:
        if isinstance(cfg.DATASETS.TRAIN, str) and "coco" in cfg.DATASETS.TRAIN:
            classes = list(range(60))
        elif isinstance(cfg.DATASETS.TRAIN, (list, tuple)) and any(["coco" in ds for ds in cfg.DATASETS.TRAIN]):
            classes = list(range(60))
        else:
            raise NotImplementedError
    elif cfg.METARCNN.PHASE == 2:
        if isinstance(cfg.DATASETS.TRAIN, str) and "coco" in cfg.DATASETS.TRAIN:
            classes = list(range(80))
        elif isinstance(cfg.DATASETS.TRAIN, (list, tuple)) and any(["coco" in ds for ds in cfg.DATASETS.TRAIN]):
            classes = list(range(80))
        else:
            raise NotImplementedError
    dataset_dicts = filter_dicts(dataset_dicts, classes)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_metarcnn_support_loader(cfg, length=100000, return_dataset=False):
    dataset_dicts = get_detection_dataset_dicts(
        cfg.METARCNN.EXTERNAL_SUPPORT_JSON,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    if cfg.METARCNN.PHASE == 1:
        if isinstance(cfg.DATASETS.TRAIN, str) and "coco" in cfg.DATASETS.TRAIN:
            classes = list(range(60))
        elif isinstance(cfg.DATASETS.TRAIN, (list, tuple)) and any(["coco" in ds for ds in cfg.DATASETS.TRAIN]):
            classes = list(range(60))
        else:
            raise NotImplementedError
    elif cfg.METARCNN.PHASE == 2:
        if isinstance(cfg.DATASETS.TRAIN, str) and "coco" in cfg.DATASETS.TRAIN:
            classes = list(range(80))
        elif isinstance(cfg.DATASETS.TRAIN, (list, tuple)) and any(["coco" in ds for ds in cfg.DATASETS.TRAIN]):
            classes = list(range(80))
        else:
            raise NotImplementedError

    dataset = DatasetFromList(dataset_dicts, copy=False)
    mapper = SupportDatasetMapper(cfg)
    dataset = SupportDataset(dataset, mapper, classes, length)

    if return_dataset:
        return dataset
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                       num_workers=0, collate_fn=trivial_batch_collator)


def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """ Change: do not remove annotations!
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def filter_dicts(dataset_dicts, classes):
    new_dataset_dicts = []
    total = len(dataset_dicts)
    cur = 0
    for data in dataset_dicts:
        append_me = True
        for anno in data["annotations"]:
            if anno["category_id"] not in classes:
                append_me = False
                break
            if is_main_process():
                log_every_n_seconds(logging.INFO,
                                    "filtering {} / {}".format(cur, total), n=5)
        if append_me:
            new_dataset_dicts.append(data)
        cur += 1
    logger = logging.getLogger(__name__)
    logger.info("{} images left after filtering".format(len(new_dataset_dicts)))
    return new_dataset_dicts
