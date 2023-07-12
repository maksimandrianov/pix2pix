import os
import time

import matplotlib.pyplot as plt


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


def display(input_img, output_img, gif_path):
    input_img = input_img.detach().cpu().numpy()
    input_img = input_img.squeeze().transpose((1, 2, 0))
    input_img = (input_img * 255).astype("uint8")
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Input")
    ax1.imshow(input_img)

    output_img = output_img.detach().cpu().numpy()
    output_img = output_img.squeeze().transpose((1, 2, 0))
    output_img = (output_img * 255).astype("uint8")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Output")
    ax2.imshow(output_img)

