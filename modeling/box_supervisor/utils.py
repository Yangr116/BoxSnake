import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold_wo_center(x, kernel_size, dilation):
    """
    -- x : shape=(N, C, H, W)

    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    ) # (N, C*k_h*k_w, L)

    size = kernel_size ** 2
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), size, x.size(2), x.size(3)
    )

    # remove the center pixels
    # size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation, sigma=2.0):
    assert images.dim() == 4

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * (1.0/sigma))

    unfolded_weights = unfold_wo_center(
        image_masks, kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights