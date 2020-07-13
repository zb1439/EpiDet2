import json
import random
import copy
import argparse
from detectron2.config import get_cfg
from detectron2.data.datasets.builtin_meta import COCO_NOVEL_CATEGORIES, COCO_CATEGORIES

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default=1, type=int)
    parser.add_argument("--shots", default=1, type=int)
    parser.add_argument("--dataset", default="coco")
    # parser.add_argument("--config-file", default="")
    return parser

def setup(args):
    cfg = get_cfg()
    cfg.freeze()
    return cfg

def make_json_file(read_path, write_path):
    phaseone = "shot" not in write_path
    json_file = json.load(open(read_path))
    num_shot = 200 if phaseone else int(write_path.split('shot')[0].split('-')[-1])
    print("preparing for {} shot support set".format(num_shot))
    all_coco_categories = [entry["id"] for entry in COCO_CATEGORIES if entry["isthing"] == 1]
    coco_novel_categories = [entry["id"] for entry in COCO_NOVEL_CATEGORIES if entry["isthing"] == 1]
    classes = [cls for cls in all_coco_categories if cls not in coco_novel_categories]\
        if phaseone else all_coco_categories
    counter = {k: 0 for k in classes}

    id_to_image_entries = {}
    for img in json_file["images"]:
        id_to_image_entries[img["id"]] = img

    images = []
    annotations = []
    for anno in json_file["annotations"]:
        if anno["area"] < 50:
           continue
        cls = anno["category_id"]
        if cls in classes and counter[cls] < num_shot:
            counter[cls] += 1
            image_id = anno["image_id"]
            image = copy.deepcopy(id_to_image_entries[image_id])
            image["id"] = anno["image_id"] = anno["id"]
            annotations.append(anno)
            images.append(image)

        done = True
        for v in counter.values():
            if v < num_shot:
                done = False
                break
        if done:
            break

    json_file["annotations"] = annotations
    json_file["images"] = images
    json_file = duplicate_shortcoming(json_file, num_shot)
    with open(write_path, 'w') as fp:
        json.dump(json_file, fp)

def duplicate_shortcoming(json_file, shots):
    bias, duplicated = 1000000, 0
    classes = set([ann['category_id'] for ann in json_file['annotations']])
    counts, class2annos = {}, {}
    for cls in classes:
        counts[cls] = 0
        class2annos[cls] = []
    id2image = {}
    for img in json_file["images"]:
        id2image[img["id"]] = img
    for ann in json_file['annotations']:
        counts[ann['category_id']] += 1
        class2annos[ann['category_id']].append(ann)

    for cls, num in counts.items():
        assert num <= shots, "category_id:{} count is over {}!".format(cls, shots)
        if num < shots:
            for i in range(shots - num):
                pickup_ann = copy.deepcopy(random.choice(class2annos[cls]))
                pickup_img = copy.deepcopy(id2image[pickup_ann["image_id"]])
                pickup_img["id"] = pickup_ann["image_id"] = pickup_ann["id"] = bias + duplicated
                json_file["annotations"].append(pickup_ann)
                json_file["images"].append(pickup_img)
                duplicated += 1
    for cls in classes:
        if counts[cls] < shots:
            print("Duplicated {} samples for category_id: {}".format(shots - counts[cls], cls))
    return json_file



if __name__ == '__main__':
    args = argument_parser().parse_args()
    cfg = setup(args)
    assert args.dataset == "coco", "only support coco right now"
    read_path = "/data/Datasets/COCO/annotations/instances_train2017.json" if args.phase == 1\
        else "/data/Datasets/COCO/finetune/{}shot_finetune.json".format(args.shots)
    write_path = "/data/Datasets/COCO/annotations/meta-rcnn-base-support.json" if args.phase == 1\
        else "/data/Datasets/COCO/finetune/meta-rcnn-{}shot-support.json".format(args.shots)
    make_json_file(read_path, write_path)