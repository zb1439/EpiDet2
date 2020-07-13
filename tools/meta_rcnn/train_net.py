#!/usr/bin/env python

import logging
import os
import time
from collections import OrderedDict
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets.coco_meta_rcnn import build_metarcnn_support_loader, \
    build_metarcnn_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOPsuedoEpisodicEvaluator,
    VOCPsuedoEpisodicEvaluator,
    COCOPrecompEpisodicEvaluator,
    VOCPrecompEpisodicEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    episodic_inference_on_dataset,
    verify_results,
    print_csv_format,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

def main_print(*msg):
    if comm.is_main_process():
        print(*msg)

def parallel_support(support, num_gpus):
    recursive_call = lambda x: parallel_support(x, num_gpus)
    if isinstance(support, torch.Tensor):
        tensor_size = [support.size(i) for i in range(support.dim())]
        return support.unsqueeze(0).expand([num_gpus] + tensor_size)
    if isinstance(support, tuple) and len(support) > 0:
        return list(zip(*map(recursive_call, support)))
    if isinstance(support, list) and len(support) > 0:
        return list(map(list, zip(*map(recursive_call, support))))
    if isinstance(support, dict) and len(support) > 0:
        return list(map(type(support), zip(*map(recursive_call, support.items()))))
    return [support for _ in range(num_gpus)]


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg, num_gpus=8):
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = build_metarcnn_train_loader(cfg)
        support_loader = build_metarcnn_support_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)
        self._support_data_iter = iter(support_loader)
        self.support_loader = support_loader
        self.num_gpus = num_gpus

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        torch.cuda.empty_cache()

        data = next(self._data_loader_iter)
        try:
            support = next(self._support_data_iter)[0]
        except:
            self._support_data_iter = iter(self.support_loader)
            support = next(self._support_data_iter)[0]
        support = parallel_support(support, self.num_gpus)

        data_time = time.perf_counter() - start

        loss_dict = self.model(data, support)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, episodic=False, ways=3):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = "coco" if "coco" in dataset_name else "voc"

        if evaluator_type == "coco":
            if not episodic:
                evaluator_list.append(COCOPsuedoEpisodicEvaluator(dataset_name, cfg, True, output_folder, way=ways))
            else:
                evaluator_list.append(COCOPrecompEpisodicEvaluator(dataset_name, cfg, True,
                                                                   output_folder, way=ways, conditioned_output=False))
        elif evaluator_type == "voc":
            if not episodic:
                evaluator_list.append(VOCPsuedoEpisodicEvaluator(dataset_name, cfg, True, output_folder, way=ways))
            else:
                evaluator_list.append(VOCPrecompEpisodicEvaluator(dataset_name, cfg, True,
                                                                   output_folder, way=ways, conditioned_output=False))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None, ways=3, shots=3, episodic=False):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = build_detection_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name, ways=ways, episodic=episodic)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            if episodic:
                attentions = type(model).load_external_attention(cfg, shots=shots)
            results_i = episodic_inference_on_dataset(model, data_loader, evaluator, attentions=attentions) \
                if episodic else inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class AttentionExtractor:
    def __init__(self, cfg):
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.logger = logger

        self.model = DefaultTrainer.build_model(cfg)
        self.support_dataset = build_metarcnn_support_loader(cfg, return_dataset=True)
        self.iters = self.support_dataset.get_real_length()
        self.cache = dict()
        self.root = cfg.METARCNN.EXTERNAL_ATTENTION_PATH + "{}shot/".format(self.iters)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        for cls in self.support_dataset.get_classes():
            self.cache[cls] = []

    def process(self):
        for it in range(self.iters):
            data = self.support_dataset[it]

            # To avoid GPU OOM, process supports by group
            num_groups = len(data) // 5 +  (0 if len(data) % 10 == 0 else 1)
            all_attentions = []
            for group in range(num_groups):
                _, support_images = self.model.preprocess_image(
                    supports=data[group * 5 : (group + 1) * 5])
                attentions = self.model.get_attentive_vector(support_images.tensor)
                all_attentions.append(attentions)
            all_attentions = torch.cat(all_attentions, dim=0)
            for sup, att in zip(data, all_attentions):
                cls = sup["class"]
                self.cache[cls].append(att.squeeze().detach().cpu().numpy())

        for k, v in self.cache.items():
            mean_attention = np.mean(v, axis=0)
            np.save(self.root + "{}.npy".format(k), mean_attention)
            self.logger.info(self.root + "{}.npy".format(k) + " saved!")

def extra_parser(parser):
    parser.add_argument(
        "--ways",
        default=3,
        type=int
    )
    parser.add_argument(
        "--shots",
        default=1,
        type=int
    )
    parser.add_argument("--episodic", action="store_true", help="episodic evaluation")
    parser.add_argument("--extract-attention", action="store_true", help="extract attention")
    return parser

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.extract_attention:
        assert args.num_gpus == 1, "only support 1 gpu for attention extraction!"
        extractor = AttentionExtractor(cfg)
        extractor.process()
        return

    if args.eval_only:
        model = Trainer.build_model(cfg)
        model.shots = args.shots
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model, ways=args.ways, shots=args.shots, episodic=args.episodic)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg, args.num_gpus)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = extra_parser(default_argument_parser()).parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
