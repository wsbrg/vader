""" Create masks based on the CIEDE2000 color distance of each frame in a video
to an image of a virtual background.
"""
import random
import colour
import click
import cv2
import multiprocessing as mp
import functools as ft
import numpy as np
from tqdm import tqdm
from pathlib import Path


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


def metric_CIEDE2000(image_1, image_2):
    image_1 = cv2.cvtColor(
        image_1.astype(np.float32) / 255, cv2.COLOR_BGR2Lab
    )
    image_2 = cv2.cvtColor(image_2.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    delta_e = colour.delta_E(image_1, image_2, method="CIE 2000")
    return delta_e


def process_video(video_path, data_path):
    deltae_path = data_path / "data" / "vbg_deltae"
    vbg_path = data_path / "virtual_backgrounds"

    name = video_path.stem
    number, gesture, bg_name, vbg_name, tool = name.split("_")
    output_path = deltae_path / name
    output_path.mkdir(exist_ok=True)

    frames = decode_video(video_path)
    vbg = cv2.imread(str(vbg_path / f"{vbg_name}.png"))
    vbg = cv2.resize(
        vbg,
        (frames[0].shape[1], frames[0].shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    for idx, frame in enumerate(frames):
        deltae = metric_CIEDE2000(frame, vbg)
        cv2.imwrite(str(output_path / f"{idx:06d}.png"), deltae)

    return True


def main(data_path, nprocs):
    vbg_videos_path = data_path / "data" / "vbg_videos"

    video_paths = list(vbg_videos_path.glob("*.mp4"))
    random.shuffle(video_paths)

    with mp.Pool(nprocs) as p:
        results = p.imap_unordered(
            ft.partial(process_video, data_path=data_path),
            video_paths,
        )
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
    "--nprocs", "-p", type=int, default=1, help="Number of parallel processes."
)
def cli(data_path, nprocs):
    main(Path(data_path), nprocs)


if __name__ == "__main__":
    cli()
