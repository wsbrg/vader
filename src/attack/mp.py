""" Re-purpose the MediaPipe virtual background feature to create portrait masks
for videos with virtual backgrounds.
"""
import click
import cv2
import numpy as np
import mediapipe
import os.path as osp
from tempfile import TemporaryDirectory
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from functools import partial


MASK_THRESHOLD = 0.64


def decode_video(input_path):
    vc = cv2.VideoCapture(str(input_path))
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        yield frame
        rval, frame = vc.read()
    vc.release()


def create_mediapipe_masks(video_path, data_path):
    target_resolution = (1280, 720)
    name, _ = osp.splitext(video_path.name)
    output_path = data_path / "data" / "vbg_mp_masks" / name
    output_path.mkdir(exist_ok=True)

    frames = decode_video(video_path)

    mp_selfie_segmentation = mediapipe.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0
    ) as selfie_segmentation:
        for i, frame in enumerate(frames):
            # Attack code
            #results = selfie_segmentation.process(
            #    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #)
            #mask = results.segmentation_mask.copy()
            #mask = mask > 0.7
            #mask = mask.astype(np.uint8) * 255
            #mask = cv2.bilateralFilter(mask, 5, 50, 50) > 0

            # BG code
            results = selfie_segmentation.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            mask = results.segmentation_mask.copy()
            mask = mask > MASK_THRESHOLD
            #mask = np.stack((mask,) * 3, axis=-1)
            #mask = (mask * 255).astype(np.uint8)
            mask = mask.astype(np.uint8) * 255

            # big blue button uses 25px wide bilateral filters
            mask = cv2.bilateralFilter(mask, 5, 25, 25)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

            frame_name = f"{i:06d}.png"
            cv2.imwrite(str(output_path / frame_name), mask)

    return True


def main(data_path, nprocs):
    vbg_videos_path = data_path / "data" / "vbg_videos"
    video_paths = vbg_videos_path.glob("*.mp4")
    video_paths = filter(lambda path: "_zoom" not in path.name, video_paths)
    video_paths = list(video_paths)

    with Pool(nprocs) as p:
        it = p.imap_unordered(partial(create_mediapipe_masks, data_path=data_path), video_paths)
        assert all(tqdm(it, total=len(video_paths)))


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
