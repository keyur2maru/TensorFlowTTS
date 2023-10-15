# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
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
"""Train ParallelWavegan."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

sys.path.append(".")

import argparse
import logging
import os
import soundfile as sf

import numpy as np
import yaml

import tensorflow_tts

#from examples.melgan.audio_mel_dataset import AudioMelDataset
#from examples.melgan.train_melgan import collater

from tensorflow_tts.configs import (
    ParallelWaveGANGeneratorConfig,
    ParallelWaveGANDiscriminatorConfig,
)
from tensorflow_tts.models import (
    TFParallelWaveGANGenerator,
    TFParallelWaveGANDiscriminator,
)

from tensorflow_tts.trainers import GanBasedTrainer
from tensorflow_tts.losses import TFMultiResolutionSTFT
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy

from tensorflow_addons.optimizers import RectifiedAdam


class ParallelWaveganTrainer(GanBasedTrainer):
    """ParallelWaveGAN Trainer class based on GanBasedTrainer."""

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
        super(ParallelWaveganTrainer, self).__init__(
            config=config,
            steps=steps,
            epochs=epochs,
            strategy=strategy,
            is_generator_mixed_precision=is_generator_mixed_precision,
            is_discriminator_mixed_precision=is_discriminator_mixed_precision,
        )

        self.list_metrics_name = [
            "adversarial_loss",
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
        self.mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae_loss = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

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
        gen_loss = 0.5 * (sc_loss + mag_loss)

        if self.steps >= self.config["discriminator_train_start_steps"]:
            p_hat = self._discriminator(y_hat)
            p = self._discriminator(tf.expand_dims(audios, 2))
            adv_loss = 0.0
            adv_loss += calculate_3d_loss(
                tf.ones_like(p_hat), p_hat, loss_fn=self.mse_loss
            )
            gen_loss += self.config["lambda_adv"] * adv_loss

            # update dict_metrics_losses
            dict_metrics_losses.update({"adversarial_loss": adv_loss})

        dict_metrics_losses.update({"gen_loss": gen_loss})
        dict_metrics_losses.update({"spectral_convergence_loss": sc_loss})
        dict_metrics_losses.update({"log_magnitude_loss": mag_loss})

        per_example_losses = gen_loss
        return per_example_losses, dict_metrics_losses

    def compute_per_example_discriminator_losses(self, batch, gen_outputs):
        audios = batch["audios"]
        y_hat = gen_outputs

        y = tf.expand_dims(audios, 2)
        p = self._discriminator(y)
        p_hat = self._discriminator(y_hat)

        real_loss = 0.0
        fake_loss = 0.0

        real_loss += calculate_3d_loss(tf.ones_like(p), p, loss_fn=self.mse_loss)
        fake_loss += calculate_3d_loss(
            tf.zeros_like(p_hat), p_hat, loss_fn=self.mse_loss
        )

        dis_loss = real_loss + fake_loss

        # calculate per_example_losses and dict_metrics_losses
        per_example_losses = dis_loss

        dict_metrics_losses = {
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "dis_loss": dis_loss,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # generate
        y_batch_ = self.one_step_predict(batch)
        y_batch = batch["audios"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            y_batch_ = y_batch_.values[0].numpy()
            y_batch = y_batch.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            y_batch_ = y_batch_.numpy()
            y_batch = y_batch.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 0):
            # convert to ndarray
            y, y_ = tf.reshape(y, [-1]).numpy(), tf.reshape(y_, [-1]).numpy()

            # plit figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavefile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                self.config["sampling_rate"],
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                self.config["sampling_rate"],
                "PCM_16",
            )


