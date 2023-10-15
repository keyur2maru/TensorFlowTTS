# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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
"""Train MelGAN Multi Resolution STFT Loss."""

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
import yaml

import tensorflow_tts
import tensorflow_tts.configs.melgan as MELGAN_CONFIG
#from examples.melgan.audio_mel_dataset import AudioMelDataset
#from examples.melgan.train_melgan import MelganTrainer, collater
from tensorflow_tts.losses import TFMultiResolutionSTFT
from tensorflow_tts.models import TFMelGANGenerator, TFMelGANMultiScaleDiscriminator
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy


class MultiSTFTMelganTrainer(MelganTrainer):
    """Multi STFT Melgan Trainer class based on MelganTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_generator_mixed_precision=False,
        is_discriminator_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_generator_mixed_precision (bool): Use mixed precision for generator or not.
            is_discriminator_mixed_precision (bool): Use mixed precision for discriminator or not.

        """
        super(MultiSTFTMelganTrainer, self).__init__(
            config=config,
            steps=steps,
            epochs=epochs,
            strategy=strategy,
            is_generator_mixed_precision=is_generator_mixed_precision,
            is_discriminator_mixed_precision=is_discriminator_mixed_precision,
        )

        self.list_metrics_name = [
            "adversarial_loss",
            "fm_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
            "spectral_convergence_loss",
            "log_magnitude_loss",
        ]

        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        super().compile(gen_model, dis_model, gen_optimizer, dis_optimizer)
        # define loss
        self.stft_loss = TFMultiResolutionSTFT(**self.config["stft_loss_params"])

    def compute_per_example_generator_losses(self, batch, outputs):
        """Compute per example generator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        dict_metrics_losses = {}
        per_example_losses = 0.0

        audios = batch["audios"]
        y_hat = outputs

        # calculate multi-resolution stft loss
        sc_loss, mag_loss = calculate_2d_loss(
            audios, tf.squeeze(y_hat, -1), self.stft_loss
        )

        # trick to prevent loss expoded here
        sc_loss = tf.where(sc_loss >= 15.0, 0.0, sc_loss)
        mag_loss = tf.where(mag_loss >= 15.0, 0.0, mag_loss)

        # compute generator loss
        gen_loss = 0.5 * (sc_loss + mag_loss)

        if self.steps >= self.config["discriminator_train_start_steps"]:
            p_hat = self._discriminator(y_hat)
            p = self._discriminator(tf.expand_dims(audios, 2))
            adv_loss = 0.0
            for i in range(len(p_hat)):
                adv_loss += calculate_3d_loss(
                    tf.ones_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
                )
            adv_loss /= i + 1

            # define feature-matching loss
            fm_loss = 0.0
            for i in range(len(p_hat)):
                for j in range(len(p_hat[i]) - 1):
                    fm_loss += calculate_3d_loss(
                        p[i][j], p_hat[i][j], loss_fn=self.mae_loss
                    )
            fm_loss /= (i + 1) * (j + 1)
            adv_loss += self.config["lambda_feat_match"] * fm_loss
            gen_loss += self.config["lambda_adv"] * adv_loss

            dict_metrics_losses.update({"adversarial_loss": adv_loss})
            dict_metrics_losses.update({"fm_loss": fm_loss})

        dict_metrics_losses.update({"gen_loss": gen_loss})
        dict_metrics_losses.update({"spectral_convergence_loss": sc_loss})
        dict_metrics_losses.update({"log_magnitude_loss": mag_loss})

        per_example_losses = gen_loss
        return per_example_losses, dict_metrics_losses


