"""
Modifications by MLCommons from SSD-Pytorch (https://github.com/uvipen/SSD-pytorch) author: Viet Nguyen (nhviet1009@gmail.com)
Copyright 2024 MLCommons Association and Contributors

MIT License

Copyright (c) 2021 Viet Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import csv
import ast

all_count = 0

class Nuscenes(Dataset):
    def __init__(self, dataset_path):
        super().__init__()

    def __len__(self):
        return 1034

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        # TODO: Load queries into memory, if needed
        pass

    def unload_query_samples(self, sample_list):
        # TODO: Unload queries from memory, if needed
        pass

    def get_samples(self, id_list):
        data = []
        labels = []
        for id in id_list:
            item = self.get_item(id)
            data.append(item)
            labels.append(None)
        return data, labels

    def get_item(self, idx):
        global all_count
        all_count += 1
        return [0]

    def get_item_count(self):
        return 1034


class PostProcessNuscenes:
    def __init__(
        self,  # Postprocess parameters
    ):
        self.content_ids = []
        # TODO: Init Postprocess parameters
        self.results = []
        self.count = 0
    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, content_id, inputs, result_dict):
        self.content_ids.extend(content_id)
        self.count += 1
        if self.count % 100 == 0:
            print("Processed {} samples".format(self.count))
        return results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):
        global all_count
        print(self.count)
        print(len(self.content_ids))
        print(all_count)

        return result_dict
