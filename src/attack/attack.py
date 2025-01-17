""" Implementation of our attack. Runing the attack requires that the virtual background
feature of the video-conferencing services was re-purposed to collect portrait masks.
For MediaPipe this is done by running `src/attack/mp.py`. For Zoom it requires the AlphaDump
tool (not published). Furthermore high quality portrait masks must be available
(`src/deeplab/deeplab.py`) as well as the color difference maps to refine the leak masks by
removing the virtual background (`src/attack/deltae.py`).
"""
import colour
import click
import os.path as osp, random
import cv2, numpy as np
import multiprocessing as mp
import functools as ft
from scipy.stats import mode
from scipy.stats import mstats
from pathlib import Path
from tqdm import tqdm


config = {
    "zoom": {
        "portrait_mask_erosion_kernel_size": 5,
        "bg_mask_erosion_kernel_size": 8,
        "min_deltae_add": 15,
        "min_deltae_sub": 5,
        # New
        "hue_range": 10,
        "sat_range": 30,
        "val_range": 40,
        "high_freq_kernel_size": 2,
        # Old
        "hsv_hue_max": 16,
        "hsv_sat_max": 205,
    },
    "mp": {
        "portrait_mask_erosion_kernel_size": 5,
        "bg_mask_erosion_kernel_size": 8,
        "min_deltae_add": 10,
        "min_deltae_sub": 4,
        # New
        "hue_range": 5,
        "sat_range": 35,
        "val_range": 40,
        "high_freq_kernel_size": 2,
        # Old
        "hsv_hue_max": 15,
        "hsv_sat_max": 210,
    },
}


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


def decode_video(path):
    """decode a video file into its frames

    Args:
        path (str): path to the video file

    Returns:
        array of uint8-matrices: frames
    """
    vc = cv2.VideoCapture(str(path))
    if vc.isOpened():
        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        raise Exception(f"Could not open video '{str(path)}'")

    frames = np.zeros((total_frames, 720, 1280, 3), dtype=np.uint8)

    i = 0
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        frame = cv2.resize(frame, (1280, 720), cv2.INTER_LINEAR)
        frames[i] = frame
        i += 1

    assert i == total_frames

    vc.release()
    return frames


def add_to_masks(masks, new_masks):
    """per-mask XOR of new_masks and masks to add new masks to the total masks

    Args:
        new_masks (array of bool-matrices): new masks to add
    """
    ones = np.ones_like(masks[0])

    for i, mask in enumerate(new_masks):
        temp = masks[i]
        temp[mask] = ones[mask]
        masks[i] = temp

    return masks


def tool_portrait_segmentation(masks, tool, zoom_path=None, mediapipe_path=None):
    """remove virtual backgrounds of all frames using mediapipe/ zoom masks

    Args:
        frames (array of uint8-matrices): input frames
        zoom (str): path to the specified zoom folder
    """
    assert (zoom_path != None) ^ (mediapipe_path != None)
    masks_path = zoom_path if zoom_path != None else mediapipe_path
    size = config[tool]["bg_mask_erosion_kernel_size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    for i, mask in enumerate(glob_masks(masks_path)):
        if size > 0:
            mask = cv2.erode(mask.astype(np.uint8), kernel, 1) > 0
        #masks[i] = ~mask # Old
        masks[i] = mask # New


def deeplab_portrait_segmentation(deeplab_path, masks, frames, tool):
    """find human pixel regions using deeplabv3_resnet101

    Args:
        frames (array of uint8-matrices): input frames

    Returns:
        human region masks as an array of uint8-matrices for use in median
    """
    # Deeplab config
    size = config[tool]["portrait_mask_erosion_kernel_size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    # Skin detection config
    lower_skin = np.array([0, 10, 40], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    hue_range = config[tool]["hue_range"]
    sat_range = config[tool]["sat_range"]
    val_range = config[tool]["val_range"]

    for i, portrait_mask_raw in enumerate(glob_masks(deeplab_path)):
        # Deeplab Segmentation
        if size > 0:
            portrait_mask = cv2.erode(portrait_mask_raw.astype(np.uint8), kernel, 1)
        masks[i][portrait_mask > 0] = False

        # Skin Detection
        frame = frames[i]
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
        skin_mask &= portrait_mask
        if np.max(skin_mask) == 0:
            continue
        median = np.median(hsv_frame[skin_mask > 0], axis=0)
        if np.any(np.isnan(median)):
            continue

        # Adaptive skin detection
        adaptive_lower_skin = np.array([
            max(int(median[0]) - hue_range, 0),
            max(int(median[1]) - sat_range, 0),
            max(int(median[2]) - val_range, 0)
        ], dtype=np.uint8)
        adaptive_upper_skin = np.array([
            min(int(median[0]) + hue_range, 179),
            min(int(median[1]) + sat_range, 255),
            min(int(median[2]) + val_range, 255)
        ], dtype=np.uint8)

        adaptive_skin_mask = cv2.inRange(hsv_frame, adaptive_lower_skin, adaptive_upper_skin)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        #adaptive_skin_mask &= cv2.dilate(portrait_mask.astype(np.uint8), kernel, 1)
        masks[i][adaptive_skin_mask > 0] = False


        #cv2.imwrite("/tmp/frame.png", frame)
        #cv2.imwrite("/tmp/ada_skin.png", adaptive_skin_mask.astype(np.uint8) * (255 / np.max(adaptive_skin_mask)))

        #import time; time.sleep(0.01)



def skin_detection(frames, masks, skin_path, name):
    for i, frame in enumerate(frames):
        # Convert the image from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define a range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        #lower_skin = np.array([0, 0, 0], dtype=np.uint8)
        #upper_skin = np.array([255, 255, 255], dtype=np.uint8)

        # Threshold the image to get a binary mask of skin color regions
        mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

        # Apply the mask to the original image
        #mask = cv2.bitwise_and(image, image, mask=skin_mask)
        masks[i] &= ~mask

#def skin_detection(frames, masks, tool):
#    """find skin pixels using a skin tone model
#
#    Args:
#        frames (array of uint8-matrices): input frames
#    """
#    hsv_hue_min = 0 # 0
#    hsv_saturation_min = 15 # 15
#    hsv_value_min = 0 # 0
#    hsv_min = (hsv_hue_min, hsv_saturation_min, hsv_value_min)
#    hsv_hue_max = config[tool]["hsv_hue_max"] # 17
#    hsv_saturation_max = config[tool]["hsv_sat_max"] # 170
#    hsv_value_max = 255 # 255
#    hsv_max = (hsv_hue_max, hsv_saturation_max, hsv_value_max)
#    for i, frame in enumerate(frames):
#        # modified from https://github.com/CHEREF-Mehdi/SkinDetection/tree/master
#        #converting from gbr to hsv color space
#        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#        #skin color range for hsv color space
#        HSV_mask = cv2.inRange(img_HSV, hsv_min, hsv_max)
#        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#
#        ##converting from gbr to YCbCr color space
#        #img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#        ##skin color range for hsv color space
#        #YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
#        #YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#
#        ##merge skin detection (YCbCr and hsv)
#        #global_mask = cv2.bitwise_and(YCrCb_mask,HSV_mask)
#        #global_mask = cv2.medianBlur(global_mask,3)
#        #global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
#
#
#        #HSV_result = cv2.bitwise_not(HSV_mask)
#        #YCrCb_result = cv2.bitwise_not(YCrCb_mask)
#        #mask = ~(cv2.bitwise_not(global_mask) > 0)
#        mask = HSV_mask > 0
#        #masks[i][mask] = True # Old
#        masks[i][mask] = False # New


def frames_pixel_mode(frames):
    # Initialize an empty array to store the modes
    modes = np.zeros(frames[0].shape, dtype=frames[0].dtype)

    # Loop through each pixel and find the mode across frames
    for i in list(range(modes.shape[0])):
        for j in range(modes.shape[1]):
            pixel_values = [frame[i, j] for frame in frames]
            modes[i, j], _ = mode(pixel_values, keepdims=False)

    return modes


def pixel_dist(p1, p2):
    return np.linalg.norm(p1-p2)

def frames_pixel_mode_low_delta(frames):
    # Initialize an empty array to store the modes
    modes = np.zeros(frames[0].shape, dtype=frames[0].dtype)
    alpha = 1.1

    # Loop through each pixel and find unique values with their counts
    for i in tqdm(list(range(frames[0].shape[0]))):
        for j in range(frames[0].shape[1]):
            pixel_values = [frame[i, j] for frame in frames]
            unique_values, counts = np.unique(pixel_values, axis=0, return_counts=True)
            distances = np.zeros(len(unique_values))

            if i > 0:
                left_adj_pixel = modes[i - 1, j]
                distances = np.array([pixel_dist(unique_pixel, left_adj_pixel) for unique_pixel in unique_values])

            # Sort values based on their counts (many to few)
            sorted_indices_freq = np.argsort(-counts)

            # Sort values based on distance to left adjacent pixel (small to huge)
            sorted_indices_dist = np.argsort(distances)
            
            # Merge the two sorts
            ranking = {}
            for rank in range(len(unique_values)):
                freq_idx = sorted_indices_freq[rank]
                dist_idx = sorted_indices_dist[rank]
                if not freq_idx in ranking:
                    ranking[freq_idx] = 0
                if not dist_idx in ranking:
                    ranking[dist_idx] = 0
                ranking[freq_idx] += rank # counts[freq_idx]
                ranking[dist_idx] += alpha * rank # alpha * distances[dist_idx]
            #ranking = sorted(ranking.items(), key=lambda el: el[1], reverse=True) # Highest (score merging)
            ranking = sorted(ranking.items(), key=lambda el: el[1]) # Lowest (rank merging)

            modes[i, j] = unique_values[ranking[0][0]]

    return modes


def frames_pixel_median(frames):
    # Initialize an empty array to store the medians
    medians = np.zeros(frames[0].shape, dtype=frames[0].dtype)

    # Loop through each pixel and find the median across frames
    for i in tqdm(list(range(medians.shape[0]))):
        for j in range(medians.shape[1]):
            pixel_values = [frame[i, j] for frame in frames]
            medians[i, j] = np.median(pixel_values)

    return medians


def metric_CIEDE2000(image_1, image_2):
    image_1 = cv2.cvtColor(
        image_1.astype(np.float32) / 255, cv2.COLOR_BGR2Lab
    )
    image_2 = cv2.cvtColor(image_2.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    delta_e = colour.delta_E(image_1, image_2, method="CIE 2000")
    return delta_e


def vbg_detection_add(vbg_deltae_path, masks, frames, tool):
    min_deltae = config[tool]["min_deltae_add"]

    reconstructed_vbg_path = Path("./modes") / f"{vbg_deltae_path.name}.png"
    if reconstructed_vbg_path.exists():
        reconstructed_vbg = cv2.imread(str(reconstructed_vbg_path))
    else:
        reconstructed_vbg = frames_pixel_mode(frames)
        cv2.imwrite(str(reconstructed_vbg_path), reconstructed_vbg)

    size = config[tool]["high_freq_kernel_size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    for idx, frame in enumerate(frames):
        deltae = cv2.imread(str(vbg_deltae_path / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        mask = deltae >= min_deltae

        mask = cv2.erode(mask.astype(np.uint8), kernel, 1) > 0
        mask = cv2.dilate(mask.astype(np.uint8), kernel, 1) > 0

        masks[idx] |= mask


def vbg_detection_sub(vbg_deltae_path, masks, frames, tool):
    min_deltae = config[tool]["min_deltae_sub"]

    reconstructed_vbg_path = Path("./modes") / f"{vbg_deltae_path.name}.png"
    if reconstructed_vbg_path.exists():
        reconstructed_vbg = cv2.imread(str(reconstructed_vbg_path))
    else:
        reconstructed_vbg = frames_pixel_mode(frames)
        cv2.imwrite(str(reconstructed_vbg_path), reconstructed_vbg)

    for idx, frame in enumerate(frames):
        deltae = cv2.imread(str(vbg_deltae_path / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        masks[idx] &= deltae >= min_deltae

        ## Test # TODO REMOVE
        #deltae = metric_CIEDE2000(frame, reconstructed_vbg)
        #masks[idx] &= deltae >= min_deltae


def frames_masked_median(frames, masks):
    # Initialize an empty array to store the medians
    medians = np.zeros(frames[0].shape, dtype=frames[0].dtype)

    # Loop through each pixel and find the mode across frames
    for i in tqdm(list(range(medians.shape[0]))):
        for j in range(medians.shape[1]):
            pixel_values = [frame[i, j] for frame in frames]
            ma = np.ma.masked_array(pixel_values, [[mask[i, j] == 0] * 3 for mask in masks])
            median = np.ma.median(ma)
            medians[i, j] = median

    return medians


def frames_masked_mode(frames):
    # Initialize an empty array to store the modes
    modes = np.zeros(frames[0].shape, dtype=frames[0].dtype)

    # Loop through each pixel and find the mode across frames
    for i in tqdm(list(range(modes.shape[0]))):
        for j in range(modes.shape[1]):
            pixel_values = [frame[i, j] for frame in frames]
            ma = np.ma.masked_array(pixel_values, [[masks[i, j] == 0] * 3 for mask in masks])
            mode = mstats.mode(ma)
            modes[i, j], _ = mode

    return modes



def create_final(frames, masks):
    """create final reconstructed image from masks

    Args:
        frames (uint8-matrix): input frames

    Returns:
        uint8-matrix: reconstructed background
    """
    leakages = np.array([np.sum(mask) for mask in masks])
    indices_sorted = np.flip(np.argsort(leakages)) # leakage: high to low
    final_image = np.zeros_like(frames[0])

    for index in indices_sorted:
        mask = masks[index]
        frame = frames[index]
        free_pixels_mask = np.all(final_image == [0, 0, 0], axis=-1)
        free_pixels_mask = np.logical_and(free_pixels_mask, mask)
        final_image[free_pixels_mask] = frame[free_pixels_mask]

    return final_image


def save_masks(path, masks):
    path.mkdir(exist_ok=True, parents=True)
    for idx, mask in enumerate(masks):
        mask_path = path / f"{idx:06d}.png"
        cv2.imwrite(str(mask_path), mask)


def reconstruct_background(video_path, data_path, opt_save_masks):
    name, _ = osp.splitext(video_path.name)
    number, gesture, bg, vbg, tool = name.split("_")

    # Zoom path
    zoom_path = None
    mediapipe_path = None
    if tool == "zoom":
        zoom_path = data_path / "data" / "vbg_zoom_masks" / name
    else:
        mediapipe_path = data_path / "data" / "vbg_mp_masks" / name

    # Deeplab masks path
    deeplab_path = data_path / "data" / "deeplab_masks" / name

    # Virtual background path
    vbg_deltae_path = data_path / "data" / "vbg_deltae" / name

    # Reconstructed frames path
    save_masks_path = data_path / "data" / "saved_masks" / "vader"

    # Reconstruct background
    frames = decode_video(video_path)

    mask_count = len(frames)
    height, width, channels = frames[0].shape
    masks = np.zeros((mask_count, height, width), dtype=np.uint8)

    # Additive steps
    tool_portrait_segmentation(masks, tool, zoom_path, mediapipe_path)
    vbg_detection_add(vbg_deltae_path, masks, frames, tool)

    # Subtractive steps
    deeplab_portrait_segmentation(deeplab_path, masks, frames, tool)
    vbg_detection_sub(vbg_deltae_path, masks, frames, tool)
    #skin_detection(frames, masks, tool)

    # Save masks
    if opt_save_masks:
        save_masks(save_masks_path / name, masks)

    final_reconstruction = create_final(frames, masks)

    # Save reconstructed background to file
    output_path = data_path / "data" / "reconstructed" / "vader"
    output_path.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path / f"{name}.png"), final_reconstruction)

    return True


def main(data_path, opt_save_masks, nprocs):
    vbg_videos_path = data_path / "data" / "vbg_videos"
    video_paths = list(vbg_videos_path.glob("*.mp4"))
    video_paths_count = len(video_paths)

    # Filter video_paths for which currently no deeplab mask exist
    video_paths = filter(
        lambda path: (
            path.parent.parent / "deeplab_masks" / osp.split(path.name)[0]
        ).exists(),
        video_paths,
    )
    video_paths = list(video_paths)

    # Make sure every video has deeplab masks
    assert video_paths_count == len(video_paths)

    random.shuffle(video_paths)

    with mp.Pool(nprocs) as p:
        reconstructed = p.imap_unordered(
            ft.partial(reconstruct_background, data_path=data_path, opt_save_masks=opt_save_masks),
            video_paths,
        )
        assert all(tqdm(reconstructed, total=len(video_paths)))


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
def cli(data_path, nprocs, save_masks):
    main(Path(data_path), save_masks, nprocs)


if __name__ == "__main__":
    cli()
