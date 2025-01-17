""" Takes the raw videos (with greenscreens) as inputs and creates ground truth
masks, replaces the greenscreen with the new "real background", and performs
the mediapipe steps replacing the "real" background with the "virtual" ones.
"""
import cv2
import numpy as np
import ffmpeg
import click
import os.path as osp
import functools as ft
from tempfile import TemporaryDirectory
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


def init_paths(data_path):
    directories = [
        "greenscreen_masks",
        "bg_mp_masks",
        "mp_leak_masks",
        "zoom_leak_masks",
        "bg_zoom_masks",
        "bg_videos",
        "vbg_videos",
        "vbg_zoom_masks",
        "vbg_mp_masks",
        "deeplab_masks",
        "reconstructed",
        "deltae",
        "ground_truths_masks",
        "vbg_deltae",
    ]
    for directory in directories:
        directory_path = data_path / "data" / directory
        directory_path.mkdir(parents=True, exist_ok=True)


def replace_background(
    frames_path,
    org_frames_path,
    gs_mask_path,
    background_path,
    target_resolution,
    number_of_frames,
):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    frame = cv2.imread(str(org_frames_path / f"{0:06d}.png"))
    background = cv2.imread(str(background_path))
    background = cv2.resize(
        background,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    for i in range(number_of_frames):
        frame_name = f"{i:06d}.png"
        frame = cv2.imread(str(org_frames_path / frame_name))

        noisy_background = generate_noise(background.copy(), mod_range=10)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(frame, (150, 20, 40), (165, 40, 50))
        frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)

        mask = cv2.erode(~mask, kernel, 1)
        mask = cv2.dilate(mask, kernel, 1)
        mask = cv2.erode(mask, kernel, 1)

        cv2.imwrite(
            str(gs_mask_path / frame_name),
            cv2.resize(mask, target_resolution, interpolation=cv2.INTER_CUBIC),
        )

        mask = cv2.GaussianBlur(mask, (5, 5), 0)  # create smoother transition region
        mask = np.dstack(
            (((255 - mask) / 255), ((255 - mask) / 255), ((255 - mask) / 255))
        )
        frame = (np.multiply(frame, 1 - mask)) + (np.multiply(noisy_background, mask))

        frame = cv2.resize(
            frame.astype("uint8"), target_resolution, interpolation=cv2.INTER_CUBIC
        )
        cv2.imwrite(str(frames_path / frame_name), frame)


def generate_noise(frame, mod_range=10):
    if mod_range == 0:
        return frame
    bg = frame.copy().astype("uint16")
    noise_map = (np.random.random(frame.shape) * mod_range).astype("uint16")
    bg += noise_map
    bg = np.where(bg > 255, 255, frame).astype("uint8")
    return bg


def decode_video(org_frames_path, input_path, number):
    vc = cv2.VideoCapture(str(input_path))
    counter = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imwrite(str(org_frames_path / f"{counter:06d}.png"), frame)
        counter += 1
        rval, frame = vc.read()
    vc.release()
    return counter


def process_raw_video(number, data_path, tmp_dir):
    gesture_lookup = {
        "standup": "g",
        "wavearm": "a",
        "wavehand": "w",
        "tilthead": "n",
        "turnhead": "h",
        "interview": "i",
    }
    target_resolution = (1280, 720)

    real_background_path = data_path / "real_backgrounds"
    real_background_paths = list(real_background_path.glob("*.png"))
    input_paths = list((data_path / "raw" / str(number)).glob("*.mp4"))

    for input_path in input_paths:
        gesture, _ = osp.splitext(input_path.name)
        gesture_abbr = gesture_lookup[gesture]

        with TemporaryDirectory(dir=tmp_dir) as tmp_directory:
            tmp_path = Path(str(tmp_directory))

            # Create temporary directory paths
            frames_path = tmp_path / "frames"
            org_frames_path = tmp_path / "org_frames"
            frames_path.mkdir(parents=True, exist_ok=True)
            org_frames_path.mkdir(parents=True, exist_ok=True)

            number_of_frames = decode_video(org_frames_path, input_path, number)

            for bg_path in real_background_paths:
                bg_name, _ = osp.splitext(bg_path.name)

                init_paths(data_path)

                # Create output directories
                gs_mask_path = (
                    data_path / "data" / "greenscreen_masks" / f"{number}_{gesture_abbr}"
                )
                mp_masks_path = (
                    data_path
                    / "data"
                    / "bg_mp_masks"
                    / f"{number}_{gesture_abbr}_{bg_name}"
                )
                mp_leak_masks_path = (
                    data_path
                    / "data"
                    / "mp_leak_masks"
                    / f"{number}_{gesture_abbr}_{bg_name}"
                )
                gs_mask_path.mkdir(parents=True, exist_ok=True)
                mp_masks_path.mkdir(parents=True, exist_ok=True)
                mp_leak_masks_path.mkdir(parents=True, exist_ok=True)

                replace_background(
                    frames_path,
                    org_frames_path,
                    gs_mask_path,
                    bg_path,
                    target_resolution,
                    number_of_frames,
                )

                ffmpeg.input(
                    str(frames_path / "%6d.png"),
                    pattern_type="sequence",
                    framerate=30,
                ).filter("fps", "30").filter("format", "yuv420p").output(
                    str(
                        data_path
                        / "data"
                        / "bg_videos"
                        / f"{number}_{gesture_abbr}_{bg_name}.mp4"
                    ),
                    **{"q:v": 0},
                ).run(
                    overwrite_output=True
                )

    return True


def main(data_path, tmp_dir, nprocs):
    raw_videos_path = data_path / "raw"
    raw_video_numbers = [path.name for path in raw_videos_path.iterdir()]
    with Pool(nprocs) as p:
        it = p.imap_unordered(
            ft.partial(process_raw_video, data_path=data_path, tmp_dir=tmp_dir), raw_video_numbers
        )
        assert all(tqdm(it, total=len(raw_video_numbers)))


@click.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--tmp-dir",
    "-t",
    default=click.Path("/tmp"),
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--nprocs",
    "-p",
    default=1,
    type=int,
)
def cli(data_path, tmp_dir, nprocs):
    main(Path(data_path), tmp_dir, nprocs)


if __name__ == "__main__":
    cli()
