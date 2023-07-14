import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

ITER_PREFIX = "__"


class LearningStats:
    def __init__(self):
        self.stats = defaultdict(list)
        self.epoch = 0

    def finish_epoch(self, need_print=True):
        self._calc_avg()
        if need_print:
            log_str = f"Epoch: {self.epoch}, "
            log_str += ", ".join(
                [f"{k}: {v[-1]}" for k, v in self.stats.items() if not k.startswith(ITER_PREFIX)]
            )
            logger.info(log_str)

        self.epoch += 1

    def push_iter_metric(self, name, value):
        self.stats[f"{ITER_PREFIX}{name}"].append(value)

    def push_metric(self, name, value):
        self.stats[name].append(value)

    def plot(self):
        plt.plot(self.stats["generator_loss_total"], color="r", label="generator_loss_total")
        plt.plot(
            self.stats["generator_loss_l1_validation"],
            color="g",
            label="generator_loss_l1_validation",
        )
        plt.xlabel("Loss")
        plt.ylabel("Epoch")
        plt.title("Learning and Validation")
        plt.legend()
        plt.show()

        plt.plot(self.stats["discriminator_loss_real"], color="r", label="discriminator_loss_real")
        plt.plot(self.stats["discriminator_loss_fake"], color="g", label="discriminator_loss_fake")
        plt.xlabel("Loss")
        plt.ylabel("Epoch")
        plt.title("Real and Fake")
        plt.legend()
        plt.show()

    def _calc_avg(self):
        for k in list(self.stats.keys()):
            if k.startswith(ITER_PREFIX):
                new_k = k.replace(ITER_PREFIX, "", 1)
                self.stats[new_k].append(np.mean(self.stats[k]))
                self.stats[k].clear()
