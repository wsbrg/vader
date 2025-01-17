import cv2
import random
import click
import functools as ft
import os.path as osp
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from pathlib import Path
from collections import defaultdict


# the paper states 0 as the threshold but that does not produce any results bc of compression or streaming artifacts
# so we used a threshold of 20 to allow for those errors
virt_bg_masking_threshold = 20


def log(msg):
    """print a message with timestamp

    Args:
        msg (str): message to log
    """
    print(msg)


def decode_video(path):
    """decode a video file into its frames

    Args:
        path (str): path to the video file

    Returns:
        array of uint8-matrices: frames
    """
    frames = []
    vc = cv2.VideoCapture(str(path))
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        frame = cv2.resize(frame, (1280, 720), cv2.INTER_LINEAR)
        frames.append(frame)
        rval, frame = vc.read()
    vc.release()
    return frames


def virtual_background_identification(shape, path):
    return cv2.resize(
        cv2.imread(str(path)),
        shape,
    )


def glob_masks(path):
    """grab all files in a directory using glob

    Args:
        path (str): path to the directory

    Returns:
        array of uint8-matrices: all png-files in the directory
    """
    mask_paths = sorted(path.glob("*.png"))
    masks = []
    for path in mask_paths:
        masks.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0)
    return masks


def virtual_background_masking(frames, virt_bg):
    masks = []
    masked_frames = []
    black = np.zeros_like(frames[0])

    for i, frame in enumerate(frames):
        # calculate distances between pixel and virtual background pixel and check if in threshold
        diff_3d = (
            np.abs(frame.astype("int32") - virt_bg.astype("int32"))
            <= virt_bg_masking_threshold
        )

        # merge three channels together
        diff = np.logical_and(
            diff_3d[:, :, 0], np.logical_and(diff_3d[:, :, 1], diff_3d[:, :, 2])
        )

        masks.append(diff)

        frame[diff] = black[diff]
        masked_frames.append(frame)

    return masks, masked_frames


def blur_masking_dilate(virt_bg_masks):
    radius = 19  # empirical value according to paper
    masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

    for i, mask in enumerate(virt_bg_masks):
        mask = cv2.dilate(mask.astype("uint8") * 255, kernel) > 0
        masks.append(mask)

    return masks


def blur_masking_colors(virt_bg_masks, frames, virt_bg):
    radius = 20  # empirical value according to paper
    masks = []

    for i, frame in enumerate(frames):
        delta_b = (
            frame[:, :, 0].astype("int32") - virt_bg[:, :, 0].astype("int32")
        ) ** 2
        delta_g = (
            frame[:, :, 1].astype("int32") - virt_bg[:, :, 1].astype("int32")
        ) ** 2
        delta_r = (
            frame[:, :, 2].astype("int32") - virt_bg[:, :, 2].astype("int32")
        ) ** 2

        mask = (delta_r + delta_g + delta_b) < (radius**2)
        masks.append(mask)

    log("applied blur masking.")
    return masks


def color_based_refinement_orig(masks, frames):
    scale = 4  # 4 default to bin pixels
    thr = 46080 / scale  # dims * 5% see paper

    for i, mask in tqdm(enumerate(masks)):
        frame = (frames[i] / scale).astype("int")
        dim = int(256 / scale) + 1
        known_colors_false = np.ones((dim, dim, dim)) == 0
        known_colors_true = np.ones((dim, dim, dim)) == 0

        # For each pixel of the mask get check the frame's color frequency
        for x in range(mask.shape[1]):
            for y in range(mask.shape[0]):
                # Skip if DeeplabV3 does think the pixel belongs to a person
                if not mask[y, x]:
                    continue

                color = frame[y, x]

                # Skip if we already know that the color's frequency is too high
                if known_colors_false[color[0], color[1], color[2]]:
                    continue

                # If we already know that the color's frequency is low, adjust the mask
                if known_colors_true[color[0], color[1], color[2]]:
                    mask[y, x] = False
                    continue

                # We do not know the color's frequency so calculate it!
                occ = np.where(frame == color, 1, 0)
                if np.sum(occ) < thr:
                    mask[y, x] = False
                    known_colors_true[
                        color[0], color[1], color[2]
                    ] = True
                else:
                    known_colors_false[
                        color[0], color[1], color[2]
                    ] = True

    return masks

def color_based_refinement_frame(masks, frames):
    assert len(frames) > 0
    assert len(masks) > 0

    min_frequency = 0.05
    height, width, _  = frames[0].shape
    pixel_count = height * width

    for i, (mask, frame) in tqdm(enumerate(zip(masks, frames))):
        pixel_values = frame.reshape((-1, 3))
        unique_colors, counts = np.unique(pixel_values, axis=0, return_counts=True)
        total = np.sum(counts)
        color_frequencies = {tuple(color): count / total for color, count in zip(unique_colors, counts)}
        x_values, y_values = np.where(mask == 1)

        for x, y in zip(x_values, y_values):
            color = frame[x][y]
            freq = color_frequencies[tuple(color)]
            if freq < min_frequency:
                mask[x][y] = False

    return masks


def calculate_color_frequencies(x, y, frames):
    color_frequencies = defaultdict(float)
    num_frames = len(frames)

    for frame in frames:
        r, g, b = frame[x][y]
        color_frequencies[(r, g, b)] += 1 / len(frames)

    return color_frequencies


def color_based_refinement_time(masks, frames):
    assert len(frames) > 0
    assert len(masks) > 0

    # 5% threshold from personal communication w/ sabra
    min_frequency = 0.05 

    # For each of these pixels, get the color frequencies and adjust the masks
    pixels = dict()

    for frame, mask in tqdm(zip(frames, masks)):
        # Find pixels belonging to the current mask
        x_values, y_values = np.where(mask == 1)

        # For each of these pixels, check the color frequency and adjust the mask
        for x, y in zip(x_values, y_values):
            # Check if we already calculated the color frequency for this pixel, if not, calculate it
            if not (x, y) in pixels:
                pixels[(x, y)] = calculate_color_frequencies(x, y, frames)

            r, g, b = frame[x][y]

            if pixels[(x, y)][(r, g, b)] < min_frequency:
                mask[x][y] = False

    return masks


def video_caller_masking(video_path, data_path):
    name, _ = osp.splitext(video_path.name)
    deeplab_path = data_path / "data" / "deeplab_masks" / name
    return glob_masks(deeplab_path)


def real_background_reconstruction(frames, masks):
    leakages = np.array([np.sum(mask) for mask in masks])
    indices_sorted = np.argsort(leakages)
    final_image = np.zeros_like(frames[0])

    for index in indices_sorted:
        mask = masks[index] == 0
        frame = frames[index]
        free_pixels_mask = np.all(final_image == [0, 0, 0], axis=-1)
        free_pixels_mask = np.logical_and(free_pixels_mask, mask)
        final_image[free_pixels_mask] = frame[free_pixels_mask]

    return final_image


def save_masks(path, masks):
    path.mkdir(exist_ok=True, parents=True)
    for idx, mask in enumerate(masks):
        mask_path = path / f"{idx:06d}.png"
        mask = mask.astype(np.uint8) * 255
        cv2.imwrite(str(mask_path), mask)


def reconstruct_background(video_path, data_path, opt_save_masks):
    name, _  = osp.splitext(video_path.name)
    number, gesture, bg_name, vbg_name, tool = name.split("_")
    vbg_path =  data_path / "virtual_backgrounds" / f"{vbg_name}.png"

    frames = decode_video(video_path)
    vbg = virtual_background_identification(
        (frames[0].shape[1], frames[0].shape[0]), vbg_path
    )
    vbg_masks, vbg_frames = virtual_background_masking(deepcopy(frames), vbg)
    bb_masks = blur_masking_dilate(vbg_masks)
    video_caller_masks = video_caller_masking(video_path, data_path)
    refined_masks = color_based_refinement_time(video_caller_masks, frames)

    masks = vbg_masks
    final_masks = []

    for i in range(len(masks) - 1):
        mask = np.logical_or(bb_masks[i], refined_masks[i])
        frames[i][mask] = 0
        final_masks.append(mask)

    # Save masks
    if opt_save_masks:
        save_masks_path = data_path / "data" / "saved_masks" / "sabra"
        save_masks(save_masks_path / name, masks)

    final_reconstruction = real_background_reconstruction(frames, final_masks)

    # Save reconstructed background to file
    output_path = data_path / "data" / "reconstructed" / "sabra"
    output_path.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path / f"{name}.png"), final_reconstruction)

    return True


def main(data_path, opt_save_masks, nprocs):
    video_paths = (data_path / "data" / "vbg_videos").glob("*.mp4")
    video_paths = list(video_paths)
    random.shuffle(video_paths) 

    with Pool(nprocs) as p:
        results = p.imap_unordered(ft.partial(reconstruct_background, data_path=data_path, opt_save_masks=opt_save_masks), video_paths)
        assert all(tqdm(results, total=len(video_paths)))


@click.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Data base directory.",
)
@click.option(
    "--save-masks/--no-save-masks",
    default=False,
)
@click.option(
    "--nprocs", "-p", type=int, default=1, help="Number of parallel processes."
)
def cli(data_path, save_masks, nprocs):
    main(Path(data_path), save_masks, nprocs)


if __name__ == "__main__":
    cli()
