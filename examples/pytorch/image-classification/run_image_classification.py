#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import evaluate
import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    softmax_exp2: bool = field(
        default=False,
        metadata={"help": "use base2 softmax in MHA (currently only support for BERT and ViT model)"},
    )
    sparsemax: bool = field(
        default=False,
        metadata={
            "help": (
                "use sparsemax in attention function (currently only support for BERT and ViT model)."
            )
        },
    )
    sparsemax_lambda: float = field(
        default=0.0,
        metadata={
            "help": (
                "use regularization factor (lambda) for sparsemax function, this is equivalent to sparsegen_lin (currently only support for BERT and ViT model)."
            )
        },
    )
    analyze_sparsity: bool = field(
        default=False,
        metadata={"help": "generate sparsity distribution report for each dot product, only works in eval mode"},
    )
    attn_sparsity_only: bool = field(
        default=False,
        metadata={"help": "dependent on --analyze_sparsity, ignore all sparsity analysis except attention scores"},
    )
    prune_attn_by_mean: bool = field(
        default=False,
        metadata={
            "help": (
                "prune attention (output of softmax) in attention function by mean filtering (currently only support for BERT and ViT model)."
            )
        },
    )
    prune_attn_by_quantile: float = field(
        default=0.0,
        metadata={
            "help": (
                "prune attention (output of softmax) by quantile (currently only support for BERT & ViT model)."
            )
        },
    )

    def __post_init__(self):
        if (float(self.softmax_exp2) + float(self.sparsemax)) > 1.0:
            raise RuntimeError("--softmax_exp2, --sparsemax are mutually exclusive")
        
        if self.sparsemax:
            if ((float(self.prune_attn_by_mean) + self.prune_attn_by_quantile)) > 0.0:
                raise RuntimeError("--prune_attn_by_quantile cannot be used with --sparsemax, --prune_attn_by_mean")
        
        if self.prune_attn_by_quantile < 0.0 or self.prune_attn_by_quantile > 1.0:
            raise RuntimeError("--prune_attn_by_quantile must be between [0, 1]")


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            task="image-classification",
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            task="image-classification",
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.softmax_exp2 = model_args.softmax_exp2
    config.use_sparsemax = model_args.sparsemax
    config.sparsemax_lambda = model_args.sparsemax_lambda
    config.prune_attn_by_mean = model_args.prune_attn_by_mean
    config.prune_attn_by_quantile = model_args.prune_attn_by_quantile
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    if training_args.do_eval and model_args.analyze_sparsity:

        if training_args.eval_batch_size > 1:
            raise ValueError("--analyze_sparsity is designed with the intent of batch size = 1")

        from collections import defaultdict, OrderedDict
        from transformers.vscutils import annotate_module_static_attr, create_sparsity_analyzer_hook, calc_sparsity_by_head
        
        per_linear_op_sparsity_dict = OrderedDict()
        per_batch_gemm_op_sparsity_dict = OrderedDict()

        annotate_module_static_attr(model, family_name="model")
        modtype_to_modlist = defaultdict(list)
        modname_to_modtype = OrderedDict()
        modname_to_module = OrderedDict()

        for _, m in model.named_modules():
            modtype_to_modlist[m.class_name].append(f"{m.last_name}.{m.first_name}")
            modname_to_modtype[m.full_name] = m.class_name
            modname_to_module[m.full_name] = m

        if model_args.attn_sparsity_only is False:
            denseoi_layers = []
            hooklist = []
            for layer in modtype_to_modlist['Linear']:
                if 'encoder' in layer:
                    denseoi_layers.append(layer)

                    mod = modname_to_module[layer]
                    hooklist.append(
                        mod.register_forward_hook(
                            create_sparsity_analyzer_hook(mod, per_linear_op_sparsity_dict)
                        )
                    )

        # NOTE: hardcoding
        attn_module_class = 'ViTSelfAttention'
        for each_attn in modtype_to_modlist[attn_module_class]:
            attn_mod = modname_to_module[each_attn]
            attn_mod.analyze_sparsity = model_args.analyze_sparsity # although it will always be true if the program gets here
            attn_mod.attn_sparsity_only = model_args.attn_sparsity_only
            attn_mod.sparsity_fn = calc_sparsity_by_head
            attn_mod.sparsity_registry = per_batch_gemm_op_sparsity_dict
            if model_args.attn_sparsity_only is False:
                attn_mod.sparsity_registry[each_attn+'.bmm1'] = defaultdict(list)
            attn_mod.sparsity_registry[each_attn+'.bmm2'] = defaultdict(list)

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if model_args.analyze_sparsity:
            import pandas as pd
            import uuid
            from functools import partial

            def check_and_return_unique_entry(row, colname):
                # assert len(pd.Series(row[colname]).unique()) == 1, "{} shape of {} must be unique".format(colname, row)
                # return pd.Series(row[colname]).unique()[0]
                return tuple(np.array(row[colname]).mean(axis=0).astype(int))

            def summarize_sparsity(row, colname):
                return_keys = ['min','max', '50%', 'mean', 'std']
                summary = pd.Series(row[colname]).describe()
                for k in return_keys:
                    row[f'{colname}_{k}'] = summary[k]
                return row

            if model_args.attn_sparsity_only is False:
                linear_df = pd.DataFrame.from_dict(per_linear_op_sparsity_dict).T
                for shape_col in ['x_shape', 'y_shape']:
                    linear_df[shape_col] = linear_df.apply(partial(check_and_return_unique_entry, colname=shape_col), axis=1)
                for sparsity_col in ['x_sparsity', 'y_sparsity']:
                    linear_df = linear_df.apply(partial(summarize_sparsity, colname=sparsity_col), axis=1).drop(columns=[sparsity_col])

            bmm_df = pd.DataFrame.from_dict(per_batch_gemm_op_sparsity_dict).T
            shape_col_list = ['x1_shape', 'x2_shape', 'y_shape']
            sparsity_col_list = ['x1_sparsity', 'x2_sparsity', 'y_sparsity']
            if model_args.attn_sparsity_only is True:
                shape_col_list = ['x1_shape']
                sparsity_col_list = ['x1_sparsity']
            for shape_col in shape_col_list:
                bmm_df[shape_col] = bmm_df.apply(partial(check_and_return_unique_entry, colname=shape_col), axis=1)
            for sparsity_col in sparsity_col_list:
                bmm_df = bmm_df.apply(partial(summarize_sparsity, colname=sparsity_col), axis=1).drop(columns=[sparsity_col])

            # Serialization
            SPARSEDIR = os.path.join(training_args.output_dir, "sparsity_analysis")
            if model_args.attn_sparsity_only is True:
                SPARSEDIR += '_attn_only' 
            uuid = str(uuid.uuid4())[:6]

            os.makedirs(SPARSEDIR, exist_ok=True)

            with open(os.path.join(SPARSEDIR, "readme.md"), "w") as f:
                f.write(f"source: {model_args.model_name_or_path}")
            with open(os.path.join(SPARSEDIR, "training_args.json"), "w") as f:
                f.write(training_args.to_json_string())

            config.to_json_file(os.path.join(SPARSEDIR, "config.json"))

            import torch
            if model_args.attn_sparsity_only is False:
                linear_df.to_csv(os.path.join(SPARSEDIR, f"{uuid}_linear_matmul_sparsity.csv"), index_label='linear_matmul')
                torch.save(per_linear_op_sparsity_dict, os.path.join(SPARSEDIR, f"{uuid}_per_linear_op_sparsity_dict.pth"))

            bmm_df.to_csv(os.path.join(SPARSEDIR, f"{uuid}_batch_matmul_sparsity.csv"), index_label='batched_matmul')
            torch.save(per_batch_gemm_op_sparsity_dict, os.path.join(SPARSEDIR, f"{uuid}_per_batch_gemm_op_sparsity_dict.pth"))

            if model_args.attn_sparsity_only is True: # only apply to attention, x1_sparsity key applies to bmm1 and bmm2
                bmm_df.x1_sparsity_mean.describe().to_csv(os.path.join(SPARSEDIR, f"{uuid}_attention_summary_over_mean_sparsity_of_all_layers.csv"))

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
