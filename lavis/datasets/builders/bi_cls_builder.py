"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.bi_cls_datasets import (
    BiCLSDataset,
    BiCLSEvalDataset,
)

from lavis.common.registry import registry



@registry.register_builder("fashion_bi_cls")
class FashionBiCLSBuilder(BaseDatasetBuilder):
    train_dataset_cls = BiCLSDataset
    eval_dataset_cls = BiCLSDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/fashion/defaults_bicls.yaml"}

