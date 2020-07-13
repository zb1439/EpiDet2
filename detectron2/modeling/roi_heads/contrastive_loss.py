import torch
import torch.nn as nn
import numpy as np
import logging
from detectron2.utils import comm

times = 0
no_ssl_times = 0
acc_dict = {}

def logical_in(a, b):
    n = a.size(0)
    m = b.size(0)
    res = torch.zeros_like(a, dtype=torch.bool)
    diff = b.unsqueeze(0).expand(n, m) - a.unsqueeze(1)
    idx = torch.where(diff == 0)[0]
    res[idx] = True
    return res

def l2_normalize(x):
    norm = torch.norm(x, p=2, keepdim=True, dim=-1)
    return x / (norm + 1e-5)

def print_rank(str, per_times=1, prefix=""):
    global times
    if comm.get_rank() == 0 and times % per_times == 0:
        print('{}: {}'.format(prefix, str))

def accuracy(logits, name):
    global times, acc_dict
    arg_max = torch.max(logits, dim=1)[1]
    acc = (arg_max == 0).sum().cpu().numpy() / (arg_max).numel()
    if name in acc_dict:
        acc_dict[name].append(acc)
    else:
        acc_dict[name] = [acc]
    acc_dict[name] = acc_dict[name][-1000:]
    print_rank(np.mean(acc_dict[name]), 200, 'acc({})'.format(name))

def main_print(*msg):
    if comm.get_rank() == 0:
        print(*msg)


class ContrastiveLoss:
    def __init__(self, cfg):
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.tau = cfg.MODEL.SSL_HEAD.TAU
        self.coeff = cfg.MODEL.SSL_HEAD.COEFF
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.logger = logging.getLogger(__name__)
        self.max_neg = cfg.MODEL.SSL_HEAD.MAX_NEG_KEYS
        self.max_pos = cfg.MODEL.SSL_HEAD.MAX_QUERIES

    @torch.no_grad()
    def _get_indices(self, proposals, targets):
        device = proposals[0].gt_classes.device

        target_classes = torch.cat([t.gt_classes for t in targets], dim=0)
        target_indices = torch.cat([t.gt_inst_index for t in targets], dim=0)
        proposal_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        proposal_indices = torch.cat([p.gt_inst_index for p in proposals], dim=0)

        gt_classes, query_indices, pos_key_indices = [], [], []
        cur_pos_num = 0
        for cls, index in zip(target_classes, target_indices):
            matched_idx = torch.nonzero(proposal_indices == index)[:, 0]
            if len(matched_idx) <= 1:
                continue
            query_idxs = matched_idx[:len(matched_idx) // 2]
            pos_key_idxs = matched_idx[-len(matched_idx) // 2:]
            if len(query_idxs) >= self.max_pos - cur_pos_num:
                query_idxs = query_idxs[:self.max_pos - cur_pos_num]
                pos_key_idxs = pos_key_idxs[:self.max_pos - cur_pos_num]
            if len(query_idxs) < len(pos_key_idxs):
                pos_key_idxs = pos_key_idxs[:len(query_idxs)]
            elif len(pos_key_idxs) < len(query_idxs):
                query_idxs = query_idxs[:len(pos_key_idxs)]

            if len(query_idxs) == 0 or len(pos_key_idxs) == 0 or \
                len(query_idxs) != len(pos_key_idxs):
                continue

            query_indices.append(query_idxs)
            pos_key_indices.append(pos_key_idxs)
            cur_pos_num += len(query_idxs)
            gt_classes.append(cls)
            if cur_pos_num >= self.max_pos:
                break

        neg_key_indices = {}
        for cls in gt_classes:
            neg = torch.nonzero(proposal_classes != cls)[:, 0]
            if len(neg) == 0:
                print("error: length of neg is 0")
                exit(0)
            neg = neg[torch.randperm(self.max_neg, device=device) % len(neg)]
            neg_key_indices[cls.item()] = neg
        return query_indices, pos_key_indices, neg_key_indices, gt_classes

    def __call__(self, origin_features, augmented_featuers, proposals, targets):
        global times
        query_indices, pos_key_indices, neg_key_indices, classes = self._get_indices(proposals, targets)
        loss = 0.0
        pairs = 0
        for query, poskey, cls in zip(query_indices, pos_key_indices, classes):
            neg_feats = l2_normalize(augmented_featuers[neg_key_indices[cls.item()]])
            query_feats = l2_normalize(origin_features[query])
            pos_feats = l2_normalize(augmented_featuers[poskey])
            pos_logits = (query_feats * pos_feats).sum(dim=1, keepdims=True)
            neg_logits = torch.einsum("nc,mc->nm", [query_feats, neg_feats])
            logits = torch.cat([pos_logits, neg_logits], dim=1) / self.tau
            labels = torch.zeros(query_feats.size(0), dtype=torch.long).to(logits.device)
            loss += self.loss(logits, labels)
            pairs += len(query_feats)

            accuracy(logits, str(cls.item()))

        if pairs == 0:
            global no_ssl_times
            no_ssl_times += 1
            if no_ssl_times % 100 == 0:
                main_print("no ssl pairs for 100 times")
            return torch.Tensor([0.0]).squeeze().to(origin_features.device)
        times += 1
        return self.coeff * loss / pairs

