# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers import ShapeSpec, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import Conv2d

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..roi_heads.meta_roi_heads import build_meta_roi_heads
from .build import META_ARCH_REGISTRY
from .rcnn import ProposalNetwork

__all__ = ["MetaRCNN"]


@META_ARCH_REGISTRY.register()
class MetaRCNN(nn.Module):
    def __init__(self, cfg, shots=3):
        super().__init__()

        self.build_backbones(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_meta_roi_heads(cfg, self.backbone.output_shape())
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.shots = shots
        self.cfg = cfg

        if self.training and cfg.METARCNN.AUXILIARY_LOSS:
            in_features = 2048 if not self.use_fpn else 256
            self.prn_cls_score = nn.Linear(in_features, self.num_classes)
            nn.init.normal_(self.prn_cls_score.weight, std=0.01)
            nn.init.constant_(self.prn_cls_score.bias, 0)
        if not self.training:
            self.attentions = MetaRCNN.load_external_attention(cfg, self.device, shots=shots)

        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.meta_backbone.conv1.parameters():
                p.requires_grad = False
            print("froze backbone params")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze rpn params")

        self.roi_heads.freeze_box(cfg)
        if cfg.MODEL.MASK_ON:
            self.roi_heads.freeze_mask(cfg)

        if cfg.METARCNN.AUXILIARY_LOSS:
            self.enable_meta_loss = True
            self.aux_cls_loss = nn.CrossEntropyLoss(reduction="mean")
            self.meta_loss_coeff = cfg.METARCNN.AUXILIARY_LOSS_WEIGHT

    def build_backbones(self, cfg, input_shape=None):
        if input_shape is None:
            input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        backbone_name = cfg.MODEL.BACKBONE.NAME
        backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
        self.backbone = backbone
        self._init_metabackbone(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs, supports=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
            supports: a list of dictionary
                (during training): keys are customized:
                    * image * annotation (same size of image, mask or box region) * class (int, 0-indexed)
                (during test): keys are customized:
                    * attention (Tensor) * class (int)

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        assert supports, "supports are only optional during inference"
        supports = supports[0]
        assert isinstance(supports[0], dict), \
            "you forgot to unzip the list of support from dataloader, supports are {}"

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        supports = self.filter_support(supports, gt_instances)
        support_classes = self.gather_classes(supports)
        query_images, support_images = self.preprocess_image(batched_inputs, supports)
        del supports

        query_features = self.backbone(query_images.tensor)

        class_attentive_vectors = self.get_attentive_vector(support_images.tensor)
        del support_images

        if self.enable_meta_loss:
            meta_loss = self.meta_loss(class_attentive_vectors, support_classes)
        else:
            meta_loss = {}

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(query_images, query_features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(query_images, query_features, proposals,
                                            class_attentive_vectors, support_classes,
                                            targets=gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(meta_loss)
        return losses

    def meta_loss(self, class_attentive_vectors, support_classes):
        class_attentive_vectors = class_attentive_vectors.squeeze(3).squeeze(2)
        scores = self.prn_cls_score(class_attentive_vectors)
        loss = self.aux_cls_loss(scores, support_classes) * self.meta_loss_coeff
        return {"meta_loss": loss}

    def filter_support(self, supports, targets):
        max_count = 8
        cur_count = 0
        filtered_supports = []
        gt_classes = torch.cat([t.gt_classes for t in targets], dim=0)
        for sup in supports:
            if sup["class"] in gt_classes:
                filtered_supports.append(sup)
                cur_count += 1
            if cur_count >= max_count:
                break
        # print(len(filtered_supports))
        return filtered_supports

    def inference(self, batched_inputs, supports=None,
                  detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if not supports and not hasattr(self, "attentions"):
            self.attentions = MetaRCNN.load_external_attention(self.cfg, self.device, shots=self.shots)
        supports = supports or self.attentions

        class_attentive_vectors = torch.stack([sup["attention"] for sup in supports], dim=0).to(self.device)
        support_classes = self.gather_classes(supports)

        images, _ = self.preprocess_image(batched_inputs=batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, class_attentive_vectors, support_classes)
        else:
            raise NotImplementedError

        if do_postprocess:
            return MetaRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs=None, supports=None):
        """
        Normalize, pad and batch the input images.
        """
        assert batched_inputs or supports
        query_images = []
        if batched_inputs:
            query_images = [x["image"].to(self.device) for x in batched_inputs]
            query_images = [(x - self.pixel_mean) / self.pixel_std for x in query_images]
            query_images = ImageList.from_tensors(query_images, self.backbone.size_divisibility)

        support_inputs = []
        if supports:
            support_images = [x["image"].to(self.device) for x in supports]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            support_annos = [x["annotation"].to(self.device) for x in supports]
            for image, anno in zip(support_images, support_annos):
                support_inputs.append(torch.cat([image, anno.unsqueeze(0)], dim=0))
            support_inputs = ImageList.from_tensors(support_inputs, self.backbone.size_divisibility)
        return query_images, support_inputs

    def gather_classes(self, supports):
        classes = [x["class"] for x in supports]
        classes = torch.LongTensor(classes).to(self.device)
        return classes

    @classmethod
    def load_external_attention(cls, cfg, device="cpu", shots=3):
        import os
        path = cfg.METARCNN.EXTERNAL_ATTENTION_PATH + "{}shot/".format(shots)
        file_names = os.listdir(path)
        vectors_and_classes = []
        for n in file_names:
            if n[-4:] == '.npy':
                file_name = cfg.METARCNN.EXTERNAL_ATTENTION_PATH + "{}shot/".format(shots) + n
                vec = torch.from_numpy(np.load(file_name)).float().to(device).unsqueeze(1).unsqueeze(2)
                cat = int(n[:-4])
                vectors_and_classes.append((vec, cat))
        vectors_and_classes.sort(key=lambda x: x[1])
        rtn = []
        for vec, cat in vectors_and_classes:
            rtn.append({"attention": vec, "class": cat})
        return rtn

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _init_metabackbone(self, cfg):
        self.meta_conv1 = Conv2d(
            4,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm("BN", 64),
        )
        self.use_fpn = "fpn" in cfg.MODEL.BACKBONE.NAME
        if self.use_fpn:
            self.fpn_mapping = nn.Linear(2048, 256)

    def get_attentive_vector(self, x):
        if self.use_fpn:
            backbone = self.backbone.bottom_up
        else:
            backbone = self.backbone

        x = self.meta_conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for stage, name in backbone.stages_and_names:
            x = stage(x)
        if x.size(1) != 2048: # res5
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = self.roi_heads.res5(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        if self.use_fpn:
            x = self.fpn_mapping(x.squeeze()).unsqueeze(2).unsqueeze(3)
        return x.sigmoid()
