import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks

from .. import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

__all__ = ["load_voc_json", "load_meta_voc_json"]

voc_cat2idx = {}

def load_voc_json(json_file, image_root, dataset_name=None, extra_annotation_keys=["segmentation"]):
    """
    :param json_file:
    :param image_root:
    :param dataset_name: w/ coco appendix, e.g. voc_2012_trainval_coco
    :param extra_annotation_keys:
    :return:
    """
    from pycocotools.coco import COCO
    print("It's coco style voc!")
    assert len(voc_cat2idx.keys()) == 0, "the voc_cat2idx should not be even initialized"
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    meta = MetadataCatalog.get(dataset_name[:-5])
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    assert min(cat_ids) == 1 and max(cat_ids) == len(cat_ids), "VOC json categories are strange?"
    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])

    for image_ix, (img_dict, anno_dict_list) in enumerate(imgs_anns):
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'
            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj["bbox_mode"] = BoxMode.XYWH_ABS

            if obj["category_id"] not in voc_cat2idx:
                voc_cat2idx[obj["category_id"]] = set()
            voc_cat2idx[obj["category_id"]].add(image_ix)

            obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def load_meta_voc_json(json_file, image_root, metadata, dataset_name, extra_annotation_keys=["segmentation"]):
    from pycocotools.coco import COCO
    assert len(voc_cat2idx.keys()) == 0, "the voc_cat2idx should not be even initialized"

    is_shots = 'shot' in dataset_name
    if is_shots:
        fileids = {}
        split_dir = os.path.join('datasets', 'cocosplit')
        if 'seed' in dataset_name:
            shot = dataset_name.split('_')[-2].split('shot')[0]
            seed = int(dataset_name.split('_seed')[-1])
            split_dir = os.path.join(split_dir, 'seed{}'.format(seed))
        else:
            shot = dataset_name.split('_')[-1].split('shot')[0]
        for idx, cls in enumerate(metadata['thing_classes']):
            json_file = os.path.join(
                split_dir, 'full_box_{}shot_{}_trainval.json'.format(shot, cls))
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
    id_map = metadata['thing_dataset_id_to_contiguous_id']

    dataset_dicts = []
    ann_keys = ['iscrowd', 'bbox', 'category_id'] + (extra_annotation_keys or [])

    if is_shots:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record['file_name'] = os.path.join(image_root,
                                                       img_dict['file_name'])
                    record['height'] = img_dict['height']
                    record['width'] = img_dict['width']
                    image_id = record['image_id'] = img_dict['id']

                    assert anno['image_id'] == image_id
                    assert anno.get('ignore', 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj['bbox_mode'] = BoxMode.XYWH_ABS
                    obj['category_id'] = id_map[obj['category_id']]
                    record['annotations'] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
    else:
        for image_ix, (img_dict, anno_dict_list) in enumerate(imgs_anns):
            record = {}
            record['file_name'] = os.path.join(image_root, img_dict['file_name'])
            record['height'] = img_dict['height']
            record['width'] = img_dict['width']
            image_id = record['image_id'] = img_dict['id']

            objs = []
            for anno in anno_dict_list:
                assert anno['image_id'] == image_id
                assert anno.get('ignore', 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj['bbox_mode'] = BoxMode.XYWH_ABS
                if obj['category_id'] in id_map:
                    if obj["category_id"] not in voc_cat2idx:
                        voc_cat2idx[obj["category_id"]] = set()
                    voc_cat2idx[obj["category_id"]].add(image_ix)

                    obj['category_id'] = id_map[obj['category_id']]
                    objs.append(obj)
            record['annotations'] = objs
            dataset_dicts.append(record)

    return dataset_dicts



