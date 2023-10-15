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
"""Train FastSpeech."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import argparse
import logging
import os
import sys

sys.path.append(".")

import numpy as np
import yaml

import tensorflow_tts
import tensorflow_tts.configs.fastspeech as FASTSPEECH_CONFIG
#from examples.fastspeech.fastspeech_dataset import CharactorDurationMelDataset
from tensorflow_tts.models import TFFastSpeech
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy


class FastSpeechTrainer(Seq2SeqBasedTrainer):
    """FastSpeech Trainer class based on Seq2SeqBasedTrainer."""

    def __init__(
        self, config, strategy, steps=0, epochs=0, is_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.

        """
        super(FastSpeechTrainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = ["duration_loss", "mel_loss_before", "mel_loss_after"]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def compute_per_example_losses(self, batch, outputs):
        """Compute per example losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        mel_before, mel_after, duration_outputs = outputs

        log_duration = tf.math.log(
            tf.cast(tf.math.add(batch["duration_gts"], 1), tf.float32)
        )
        duration_loss = self.mse(log_duration, duration_outputs)
        mel_loss_before = calculate_3d_loss(batch["mel_gts"], mel_before, self.mae)
        mel_loss_after = calculate_3d_loss(batch["mel_gts"], mel_after, self.mae)

        per_example_losses = duration_loss + mel_loss_before + mel_loss_after

        dict_metrics_losses = {
            "duration_loss": duration_loss,
            "mel_loss_before": mel_loss_before,
            "mel_loss_after": mel_loss_after,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function.
        outputs = self.one_step_predict(batch)

        mels_before, mels_after, *_ = outputs
        mel_gts = batch["mel_gts"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = mels_before.values[0].numpy()
            mels_after = mels_after.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            mels_before = mels_before.numpy()
            mels_after = mels_after.numpy()
            mel_gts = mel_gts.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_before, mel_after) in enumerate(
            zip(mel_gts, mels_before, mels_after), 0
        ):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_before = tf.reshape(mel_before, (-1, 80)).numpy()  # [length, 80]
            mel_after = tf.reshape(mel_after, (-1, 80)).numpy()  # [length, 80]

            # plit figure and save it
            utt_id = utt_ids[idx].decode("utf-8")
            figname = os.path.join(dirname, f"{utt_id}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title("Predicted Mel-before-Spectrogram")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title("Predicted Mel-after-Spectrogram")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()


