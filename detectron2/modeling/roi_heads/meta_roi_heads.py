import torch
import torch.nn as nn
import torch.nn.functional as F
from .roi_heads import ROIHeads, Res5ROIHeads, StandardROIHeads, \
    select_foreground_proposals, select_proposals_with_visible_keypoints
from detectron2.utils.registry import Registry
from detectron2.structures import Boxes

META_ROI_HEADS_REGISTRY = Registry("META_ROI_HEADS")
META_ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a Episodic R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""
def build_meta_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return META_ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def aggregation(all_scores, all_box_deltas, num_classes, cls_agnostic_box=False):
    # To make sure the implementation is the same as original (https://github.com/yanxp/MetaR-CNN),
    # background attributes are selected by the first class conditioned output
    scores, deltas = [], []
    for ix, (score, delta) in enumerate(zip(all_scores, all_box_deltas)):
        scores.append(score[:, ix].unsqueeze(1))
        if cls_agnostic_box:
            # There is no real cls agnostic box for meta r-cnn
            deltas.append(delta)
        else:
            deltas.append(delta[:, ix * 4: (ix + 1) * 4])
    scores.append(all_scores[0][:, num_classes].unsqueeze(1))
    scores = torch.cat(scores, dim=1)
    deltas = torch.cat(deltas, dim=1)
    return scores, deltas

@META_ROI_HEADS_REGISTRY.register()
class MetaRes5ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super(MetaRes5ROIHeads, self).__init__(cfg, input_shape)
        self.cls_agnostic_box = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK

    def forward(self, images, features, proposals,
                class_attentive_vectors, support_classes, targets=None):
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        query_box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        if self.training:
            del features
            loss_dicts = []
            proposal_labels = torch.unique(torch.cat([p.gt_classes for p in proposals], dim=0))
            for cls, attention in zip(support_classes, class_attentive_vectors):
                if torch.nonzero(proposal_labels == cls).numel() == 0:
                    continue
                conditioned_box_features = query_box_features * attention.unsqueeze(0)
                conditioned_predictions = self.box_predictor(conditioned_box_features.mean(dim=[2, 3]))

                losses = self.box_predictor.losses(conditioned_predictions, proposals)
                if self.mask_on:
                    mask_proposals, fg_selection_masks = select_foreground_proposals(
                        proposals, self.num_classes
                    )
                    conditioned_mask_features = conditioned_box_features[torch.cat(fg_selection_masks, dim=0)]
                    del conditioned_box_features
                    losses.update(self.mask_head(conditioned_mask_features, mask_proposals))
                loss_dicts.append(losses)

            assert len(loss_dicts) > 0
            mean_loss = {}
            for k in loss_dicts[0].keys():
                mean_loss[k] = sum([loss[k] for loss in loss_dicts]) / len(loss_dicts)
            return [], mean_loss
        else:
            all_preds = []
            for cls, attention in zip(support_classes, class_attentive_vectors):
                conditioned_box_features = query_box_features * attention.unsqueeze(0)
                conditioned_predictions = self.box_predictor(conditioned_box_features.mean(dim=[2, 3]))
                all_preds.append(conditioned_predictions)
            all_scores, all_box_deltas = list(zip(*all_preds))
            predictions = aggregation(all_scores, all_box_deltas, self.num_classes, self.cls_agnostic_box)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances, class_attentive_vectors)
            return pred_instances, {}


    def forward_with_given_boxes(self, features, instances, attentions):
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            query_features = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            pred_classes = torch.cat([i.pred_classes for i in instances], dim=0)
            conditioned_features = query_features * attentions[pred_classes]
            return self.mask_head(conditioned_features, instances)
        else:
            return instances

@META_ROI_HEADS_REGISTRY.register()
class MetaStandardROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super(MetaStandardROIHeads, self).__init__(cfg, input_shape)
        self.cls_agnostic_box = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BOX_REG
        self.cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK

    def forward(self, images, features, proposals,
                class_attentive_vectors, support_classes, targets=None):
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        proposal_labels = torch.unique(torch.cat([p.gt_classes for p in proposals], dim=0))

        box_features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(box_features, [x.proposal_boxes for x in proposals])

        if self.mask_on and self.training:
            mask_features = [features[f] for f in self.mask_in_features]
            fg_proposals, _ = select_foreground_proposals(proposals, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in fg_proposals]
            mask_features = self.mask_pooler(mask_features, proposal_boxes)
        else:
            mask_features = None

        if self.keypoint_on and self.training:
            keypoint_features = [features[f] for f in self.keypoint_in_features]
            fg_proposals, _ = select_foreground_proposals(proposals, self.num_classes)
            fg_proposals = select_proposals_with_visible_keypoints(fg_proposals)
            proposal_boxes = [x.proposal_boxes for x in fg_proposals]
            keypoint_features = self.mask_pooler(keypoint_features, proposal_boxes)
        else:
            keypoint_features = None

        if self.training:
            del features
            loss_dicts = []
            for cls, attention in zip(support_classes, class_attentive_vectors):
                if torch.nonzero(proposal_labels == cls).numel() == 0:
                    continue
                losses = self._forward_box(box_features, proposals, attention.unsqueeze(0))
                losses.update(self._forward_mask(mask_features, proposals, attention.unsqueeze(0)))
                losses.update(self._forward_keypoint(keypoint_features, proposals, attention.unsqueeze(0)))
                loss_dicts.append(losses)

            assert len(loss_dicts) > 0
            mean_loss = {}
            for k in loss_dicts[0].keys():
                mean_loss[k] = sum([loss[k] for loss in loss_dicts]) / len(loss_dicts)
            return proposals, mean_loss
        else:
            all_preds = []
            for cls, attention in zip(support_classes, class_attentive_vectors):
                all_preds.append(self._forward_box(box_features, proposals, attention.unsqueeze(0)))
            all_scores, all_box_deltas = list(zip(*all_preds))
            predictions = aggregation(all_scores, all_box_deltas, self.num_classes, self.cls_agnostic_box)

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances, class_attentive_vectors)
            return pred_instances, {}

    def _forward_box(self, features, proposals, attention):
        """ Change: 1.returns scores and box deltas during inference here.
                    2.features are already roi aligned.
        """
        conditioned_box_features = features * attention
        conditioned_box_features = self.box_head(conditioned_box_features)
        predictions = self.box_predictor(conditioned_box_features)
        del conditioned_box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            return predictions

    def _forward_mask(self, features, instances, attention):
        """ Change: 1.argument attention is different during training and inference.
                    2.features are already roi aligned.
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            return self.mask_head(features * attention, instances)
        else:
            assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
            pred_classes = torch.cat([i.pred_classes for i in instances], dim=0)
            return self.mask_head(features * attention[pred_classes], instances)

    def _forward_keypoint(self, features, instances, attention):
        """ Change: 1.argument attention is different during training and inference.
                    2.features are already roi aligned.
        """
        if not self.keypoint_on:
            return {} if self.training else instances
        if self.training:
            return self.keypoint_head(features * attention, instances)
        else:
            assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
            pred_classes = torch.cat([i.pred_classes for i in instances], dim=0)
            return self.keypoint_head(features * attention[pred_classes], instances)

    def forward_with_given_boxes(self, features, instances, attentions):
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        pred_boxes = [x.pred_boxes for x in instances]
        mask_features = self.mask_pooler(features, pred_boxes) if self.mask_on else None
        keypoint_features = self.keypoint_pooler(features, pred_boxes) if self.keypoint_on else None

        instances = self._forward_mask(mask_features, instances, attentions)
        instances = self._forward_keypoint(keypoint_features, instances, attentions)
        return instances
