import logging
import math
import os
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from net.blocks import Discriminator, Generator
from net.dataloader import Dataset, Direction, Transformer
from net.utils import display, set_grad

logger = logging.getLogger(__name__)

CPU_COUNT = min(1, os.cpu_count())
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_FILENAME = "weight.tar"
FILTERS = 64
IN_CHANNELS = 3
OUT_CHANNELS = 3


class Pix2PixTrainer:
    def __init__(
        self,
        root_data_dir,
        dataset_name,
        direction: Direction,
        weight_path,
        debug_mode=False,
        batch_size=16,
    ):
        self.debug_mode = debug_mode
        self.root_data_dir = root_data_dir
        self.dataset_name = dataset_name
        self.direction = direction
        self.batch_size = batch_size
        self.weight_path = weight_path

        self.train_loader = self._make_data_loader(
            "train", self.batch_size * 10 if debug_mode else None
        )
        self.val_loader = self._make_data_loader(
            "val", self.batch_size * 2 if debug_mode else None
        )

        self.generator = Generator(IN_CHANNELS, OUT_CHANNELS, FILTERS).to(DEV)
        self.discriminator = Discriminator(IN_CHANNELS + OUT_CHANNELS, FILTERS).to(DEV)

        self.learning_rate = 2e-5
        self.L1_loss = nn.L1Loss()
        self.l1_weight = 100
        self.best_v_loss = math.inf
        self.loss_discriminator = None
        self.loss_generator = None
        self.opt_discriminator = None
        self.opt_generator = None
        self._optimization_init()

        self.stats = defaultdict(list)
        self.last_epoch = 0

    def train(self, epochs=10, need_print=False):
        for epoch in range(self.last_epoch, self.last_epoch + epochs):
            self.generator.train()
            self.discriminator.train()
            for i, batch in enumerate(self.train_loader):
                input = batch["input"].to(DEV)
                target = batch["target"].to(DEV)

                fake_output = self.generator(input)
                self._step_discriminator(input, fake_output, target, i)
                self._step_generator(input, fake_output, target)
                torch.cuda.empty_cache()

            logger.info(
                f"Train.. Epoch: {epoch}, last L1 loss:"
                f" {self.stats['l1'][-1] / self.l1_weight}, last loss real:"
                f" {self.stats['loss_discriminator_real'][-1]}, last loss fake:"
                f" {self.stats['loss_discriminator_fake'][-1]}"
            )

            self._validate(epoch, need_print)
            self.last_epoch = epoch + 1

    def _save(self, state):
        filename = os.path.join(self.weight_path, WEIGHT_FILENAME)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path, exist_ok=True)

        logger.info(f"Saving weights to {filename}..")
        torch.save(state, filename)

    def _make_data_loader(self, mode, max_items):
        dataset = Dataset(
            root_dir=self.root_data_dir,
            dataset=self.dataset_name,
            mode=mode,
            direction=self.direction,
            max_items=max_items,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=CPU_COUNT,
            drop_last=True,
        )

    def _optimization_init(self):
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )

        self.loss_generator = nn.BCELoss().to(DEV)
        self.loss_discriminator = nn.BCELoss().to(DEV)

    def _step_discriminator(self, input, fake_output, target, i):
        set_grad(self.discriminator, True)
        self.opt_discriminator.zero_grad()

        fake_batch = torch.cat((input, fake_output), 1)
        fake_batch = fake_batch.detach()
        pred_fake = self.discriminator(fake_batch)
        loss_discriminator_fake = self.loss_discriminator(pred_fake, torch.ones_like(pred_fake))

        real_batch = torch.cat((input, target), 1)
        pred_real = self.discriminator(real_batch)
        loss_discriminator_real = self.loss_discriminator(pred_real, torch.zeros_like(pred_real))

        loss_discriminator_total = 0.5 * (loss_discriminator_fake + loss_discriminator_real)
        loss_discriminator_total.backward()
        self.opt_discriminator.step()

        self.stats["loss_discriminator_fake"].append(loss_discriminator_fake.item())
        self.stats["loss_discriminator_real"].append(loss_discriminator_real.item())
        self.stats["loss_discriminator_total"].append(loss_discriminator_total.item())

    def _step_generator(self, input, fake_output, target):
        set_grad(self.discriminator, False)
        self.opt_generator.zero_grad()

        fake_batch = torch.cat((input, fake_output), 1)
        pred_fake = self.discriminator(fake_batch)
        loss_generator_GAN = self.loss_generator(pred_fake, torch.ones_like(pred_fake))
        l1 = self.L1_loss(fake_output, target) * self.l1_weight
        loss_generator = l1 + loss_generator_GAN

        loss_generator.backward()
        self.opt_generator.step()

        self.stats["l1"].append(l1.item())
        self.stats["loss_generator_GAN"].append(loss_generator_GAN.item())
        self.stats["loss_generator_total"].append(loss_generator.item())

    def _validate(self, epoch, need_print):
        loss = 0
        self.generator.eval()
        for i, batch in enumerate(self.val_loader):
            input = batch["input"].to(DEV)
            target = batch["target"].to(DEV)
            with torch.no_grad():
                output = self.generator(input)

            val_loss = self.L1_loss(output, target)
            loss += val_loss

            logger.debug(f"Validate.. Epoch: {epoch}, iter: {i}, L1 loss: {val_loss.item()}")
            if need_print and i == 0:
                path = os.path.join(self.weight_path, "images", f"{epoch}.png")
                display(input[0], output[0], target[0], path)

        loss_avg = loss.item() / len(self.val_loader)
        self.stats["loss_validation"].append(loss_avg)
        logger.info(f"Validate.. Epoch: {epoch}, epoch loss: {loss_avg}")
        if loss < self.best_v_loss:
            self.best_v_loss = loss
            self._save(self.generator.state_dict())


class Pix2Pix:
    def __init__(
        self,
        root_data_dir,
        dataset_name,
        direction: Direction,
        weight_path,
        debug_mode,
    ):
        self.debug_mode = debug_mode
        self.root_data_dir = root_data_dir
        self.dataset_name = dataset_name
        self.direction = direction
        self.weight_path = weight_path
        self.test_loader = self._dataset_init(debug_mode)

        self.generator = Generator(IN_CHANNELS, OUT_CHANNELS, FILTERS)
        self._load(self.generator)
        self.generator.to(DEV)

    def test(self, need_display=False):
        L1_loss = nn.L1Loss()
        t_loss = 0
        self.generator.eval()
        for i, batch in enumerate(self.test_loader):
            input = batch["input"].to(DEV)
            target = batch["target"].to(DEV)
            with torch.no_grad():
                output = self.generator(input)

            test_loss = L1_loss(output, target)
            t_loss += test_loss
            if need_display:
                display(input, output, target)

            logger.debug(f"Test.. Iter: {i}, L1 loss: {test_loss.item()}")
        logger.info(f"Test.. Loss: {t_loss.item() / len(self.test_loader)}")

    def transfer_style(self, image_path, need_display=False):
        transformer = Transformer(self.debug_mode)
        input = transformer(image_path).to(DEV)
        with torch.no_grad():
            output = self.generator(input)

        if need_display:
            display(input, output)

    def _dataset_init(self, debug_mode):
        test_dataset = Dataset(
            root_dir=self.root_data_dir,
            dataset=self.dataset_name,
            mode="test",
            direction=self.direction,
            debug_mode=debug_mode,
        )
        return DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=CPU_COUNT)

    def _load(self, generator):
        filename = os.path.join(self.weight_path, WEIGHT_FILENAME)
        logger.info(f"Loading weights from {filename}..")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File does not exist {filename}")

        checkpoint = torch.load(filename, map_location=DEV)
        generator.load_state_dict(checkpoint)
        return checkpoint
