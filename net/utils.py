import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_files(path, prefix):
    return sorted(
        [os.path.join(path, file) for file in os.listdir(path) if file.startswith(prefix)]
    )


def set_grad(nets, grad):
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is None:
            continue

        for param in net.parameters():
            param.requires_grad = grad


def rescale(data, new_min=0, new_max=1):
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        data = np.array(data)
    return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min


def display(input_img, output_img, target_img=None, path=None, epoch=None, only_save=False):
    input_img = input_img.detach().cpu().numpy()
    input_img = input_img.squeeze().transpose((1, 2, 0))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Input")
    input_img = input_img * 0.5 + 0.5
    ax1.imshow(input_img)

    output_img = output_img.detach().cpu().numpy()
    output_img = output_img.squeeze().transpose((1, 2, 0))
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Output")
    output_img = output_img * 0.5 + 0.5
    ax2.imshow(output_img)

    if target_img is not None:
        target_img = target_img.detach().cpu().numpy()
        target_img = target_img.squeeze().transpose((1, 2, 0))
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("Target")
        target_img = target_img * 0.5 + 0.5
        ax3.imshow(target_img)

    if epoch is not None:
        fig.suptitle(f"Epoch {epoch}")

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)

    if only_save:
        plt.close(fig)
    else:
        plt.show()
