import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from net.utils import rescale

logger = logging.getLogger(__name__)

ITER_PREFIX = "__"

plt.rc("lines", linewidth=4)
plt.rc(
    "axes",
    prop_cycle=(
        cycler("color", ["r", "g", "b", "y"]) + cycler("linestyle", ["-", "--", ":", "-."])
    ),
)


class LearningStats:
    def __init__(self):
        self.stats = defaultdict(list)
        self.epoch = 0

    def finish_epoch(self, need_print=True):
        self._calc_avg()
        if need_print and self.epoch % 5 == 0:
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
        self._plot()
        epochs = len(self.stats["generator_loss_l1"])
        if epochs > 70:
            self._plot(60)

    def _plot(self, from_=0):
        epochs = len(self.stats["generator_loss_l1"])
        indexes = range(from_, epochs)
        for k, v in self.stats.items():
            if k.startswith("generator"):
                plt.plot(
                    indexes,
                    rescale(v[from_:]),
                    label=k,
                )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Generator metrics {from_}-{epochs}")
        plt.legend()
        plt.show()

        for k, v in self.stats.items():
            if k.startswith("discriminator"):
                plt.plot(
                    indexes,
                    rescale(v[from_:]),
                    label=k,
                )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Discriminator {from_}-{epochs}")
        plt.legend()
        plt.show()

    def _calc_avg(self):
        for k in list(self.stats.keys()):
            if k.startswith(ITER_PREFIX):
                new_k = k.replace(ITER_PREFIX, "", 1)
                self.stats[new_k].append(np.mean(self.stats[k]))
                self.stats[k] = []
