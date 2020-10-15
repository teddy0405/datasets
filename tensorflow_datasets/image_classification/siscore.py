# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""SI-SCORE synthetic dataset."""

import os

import tensorflow.compat.v2 as tf
from tensorflow_datasets.image_classification import siscore_labels
import tensorflow_datasets.public_api as tfds


_CITATION = """\
@misc{djolonga2020robustness,
      title={On Robustness and Transferability of Convolutional Neural Networks}, 
      author={Josip Djolonga and Jessica Yung and Michael Tschannen and Rob Romijnders and Lucas Beyer and Alexander Kolesnikov and Joan Puigcerver and Matthias Minderer and Alexander D'Amour and Dan Moldovan and Sylvain Gelly and Neil Houlsby and Xiaohua Zhai and Mario Lucic},
      year={2020},
      eprint={2007.08558},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DESCRIPTION = """
SI-Score (Synthetic Interventions on Scenes for Robustness Evaluation) is a 
dataset to evaluate robustness of image classification models to changes in 
object size, location and rotation angle.

In SI-SCORE, we take objects and backgrounds and systematically vary object 
size, location and rotation angle so we can study the effect of changing these 
factors on model performance. The image label space is the ImageNet
label space (1k classes) for easy evaluation of models.

More information about the dataset can be found at https://github.com/google-research/si-score.
"""

_NUM_CLASSES = 61

_BASE_URL = "https://s3.us-east-1.amazonaws.com/si-score-dataset"

_VERSION = tfds.core.Version("0.2.0")


class SiscoreConfig(tfds.core.BuilderConfig):
  """BuilderConfig for SI-Score."""

  def __init__(self, *, variant, **kwargs):
    """BuilderConfig for SI-Score.

    Args:
      variant: str. The synthetic dataset variant. One of 'rotation', 'size'
        and 'location'.
      **kwargs: keyword arguments forwarded to super.
    """
    super(SiscoreConfig, self).__init__(
        **kwargs)
    self.variant = variant


class Siscore(tfds.core.GeneratorBasedBuilder):
  """SI-Score synthetic image dataset."""

  BUILDER_CONFIGS = [
      SiscoreConfig(
          variant=x, name=x, version=_VERSION, description=_DESCRIPTION)
      for x in ["rotation", "size", "location"]
  ]

  def _info(self):
    # TODO(jessicayung): add version?
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            "image_id": tf.int64,
            "image": tfds.features.Image(),
            # ImageNet label space
            "label": tfds.features.ClassLabel(num_classes=1000),
            "dataset_label": tfds.features.ClassLabel(num_classes=_NUM_CLASSES),
        }),
        supervised_keys=("image", "label"),
        # Homepage of the dataset for documentation
        homepage="https://github.com/google-research/si-score",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerator."""
    # using rotation link only for now
    variant = self.builder_config.variant
    dataset_url = os.path.join(_BASE_URL, f"{variant}.zip")
    # path = dl_manager.download_and_extract(dataset_url)
    path = os.path.join(dl_manager.download_and_extract(dataset_url), variant)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={"datapath": path},
        ),
    ]

  def _generate_examples(self, datapath):
    """Yields examples of synthetic data images and labels."""
    for fpath in tf.io.gfile.glob(os.path.join(datapath, "*/*.jpg")):
      label = fpath.split("/")[-2]
      fname = os.path.basename(fpath)
      record = {
          "image": fpath,
          "image_id": int(fname.split(".")[0]),
          "label": siscore_labels.imagenet_labels[label],
          "dataset_label": siscore_labels.dataset_labels[label],
      }
      yield fname, record
