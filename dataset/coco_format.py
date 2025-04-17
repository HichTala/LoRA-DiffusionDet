# Modified by HichTala

# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
# limitations under the License.
"""COCO"""
import json
import os
from pathlib import Path

import datasets


class COCO(datasets.GeneratorBasedBuilder):
    """COCO"""

    VERSION = None

    BUILDER_CONFIGS = []

    DEFAULT_CONFIG_NAME = ""

    def __init__(self, name, annotations_path, image_path, features, version=datasets.Version("1.0.0")):
        self.VERSION = version

        self.BUILDER_CONFIGS.append(datasets.BuilderConfig(
            name=name, version=self.VERSION, description=""
        ))

        self.DEFAULT_CONFIG_NAME = name

        self.features = datasets.Features(features)

        super().__init__()

        self.annotations_path = annotations_path
        self.image_path = image_path

    def _info(self):
        return datasets.DatasetInfo(
            features=self.features,
        )

    def _split_generators(self, dl_manager):
        annotation_file = {
            "train": os.path.join(self.annotations_path, "instances_train2017.json"),
            "validation": os.path.join(self.annotations_path, "instances_val2017.json"),
            "test": os.path.join(self.annotations_path, "instances_test2017.json")}
        image_folders = {"train": os.path.join(self.image_path, "train2017"),
                         "validation": os.path.join(self.image_path, "val2017"),
                         "test": os.path.join(self.image_path, "test2017")}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": annotation_file["train"],
                    "image_folders": image_folders,
                    "split_key": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": annotation_file["validation"],
                    "image_folders": image_folders,
                    "split_key": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file": annotation_file["test"],
                    "image_folders": image_folders,
                    "split_key": "test",
                },
            ),
        ]

    def _generate_examples(self, annotation_file, image_folders, split_key):
        with open(annotation_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

            for image_metadata in annotations["images"]:
                image_path = os.path.join(image_folders[split_key], image_metadata["file_name"])

                record = {
                    "image_id": image_metadata["id"],
                    "image": str(Path(image_path).absolute()),
                    "width": image_metadata["width"],
                    "height": image_metadata["height"],
                    "objects": [{
                        "bbox_id": ann["id"],
                        "category": ann["category_id"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                    } for ann in annotations["annotations"] if ann["image_id"] == image_metadata["id"]]
                }
                yield record["image_id"], record
