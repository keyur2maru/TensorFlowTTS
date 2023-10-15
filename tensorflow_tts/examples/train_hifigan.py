# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team
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
"""Train Hifigan."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

import tensorflow_tts
#from examples.melgan.audio_mel_dataset import AudioMelDataset
#from examples.melgan.train_melgan import collater
#from examples.melgan_stft.train_melgan_stft import MultiSTFTMelganTrainer
from tensorflow_tts.configs import (
    HifiGANDiscriminatorConfig,
    HifiGANGeneratorConfig,
    MelGANDiscriminatorConfig,
)
from tensorflow_tts.models import (
    TFHifiGANGenerator,
    TFHifiGANMultiPeriodDiscriminator,
    TFMelGANMultiScaleDiscriminator,
)
from tensorflow_tts.utils import return_strategy


class TFHifiGANDiscriminator(tf.keras.Model):
    def __init__(self, multiperiod_dis, multiscale_dis, **kwargs):
        super().__init__(**kwargs)
        self.multiperiod_dis = multiperiod_dis
        self.multiscale_dis = multiscale_dis

    def call(self, x):
        outs = []
        period_outs = self.multiperiod_dis(x)
        scale_outs = self.multiscale_dis(x)
        outs.extend(period_outs)
        outs.extend(scale_outs)
        return outs

