import cv2
import click
import numpy as np
import mediapipe as mp
import functools as ft
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def virtual_background(
    gs_mask_path,
    zoom_masks_path,
    vbg_frames_path,
    frames,
    virt_masks,
    virtual_background_path,
    target_resolution,
):
    background = cv2.imread(str(virtual_background_path))
    background = cv2.resize(
        background,
        (frames[0].shape[1], frames[0].shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    for i, frame in enumerate(frames):
        mask_slice = virt_masks[i].copy()
        temp_mask = (255 - virt_masks[i]) / 255
        temp_mask = np.stack((temp_mask,) * 3, axis=-1)

        frame = (np.multiply(frame, 1 - temp_mask)) + (
            np.multiply(background, temp_mask)
        )
        cv2.imwrite(str(vbg_frames_path / f"{i:06d}.png"), frame)

        # create mp_leak_masks aka bg_mp_masks - gs_mask
        gs_mask = cv2.imread(str(gs_mask_path / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        gs_mask = cv2.resize(
            gs_mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        gs_mask = gs_mask > 0
        black = np.zeros_like(mask_slice)

        mask_slice[gs_mask] = black[gs_mask]
        cv2.imwrite(
            str(zoom_masks_path / f"{i:06d}.png"),
            cv2.resize(mask_slice, target_resolution, interpolation=cv2.INTER_CUBIC),
        )


def read_masks(path):
    paths = sorted(path.glob("*.png"))
    masks = []
    for p in paths:
        masks.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    return masks


def decode_video(path):
    vc = cv2.VideoCapture(str(path))
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    org_frames = []
    while rval:
        org_frames.append(frame)
        rval, frame = vc.read()
    vc.release()
    return org_frames


def create_vbg_masks(
    frames,
    mp_masks_path,
    virtual_background_path,
    target_resolution,
):
    background = cv2.imread(str(virtual_background_path))
    background = cv2.resize(
        background,
        (frames[0].shape[1], frames[0].shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0
    ) as selfie_segmentation:
        for i, frame in enumerate(frames):
            frame_name = f"{i:06d}.png"
            results = selfie_segmentation.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            mask = results.segmentation_mask.copy()
            mask = np.stack((mask,) * 3, axis=-1)
            mask = (mask * 255).astype(np.uint8)

            # big blue button uses 25px wide bilateral filters
            # create mp_masks
            mask = cv2.bilateralFilter(mask, 5, 25, 25)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            mask_slice = mask.copy()[:, :, 0]
            cv2.imwrite(
                str(mp_masks_path / frame_name),
                cv2.resize(mask, target_resolution, interpolation=cv2.INTER_CUBIC),
            )


def process_video(video_path, data_path):
    virtual_background_path = data_path / "virtual_backgrounds"
    virtual_background_paths = list(virtual_background_path.glob("*.png"))
    target_resolution = (1280, 720)

    frames = decode_video(video_path)
    name = video_path.stem
    number, gesture, bg = name.split("_")

    mp_masks_path = data_path / "data" / "bg_mp_masks" / name
    mp_masks_path.mkdir(exist_ok=True)

    # TODO: REMOVE
    #zoom_masks_path = data_path / "data" / "bg_zoom_masks" / f"{number}_{gesture}_{bg}"
    #masks = read_masks(zoom_masks_path)

    # create mediapipe virtual backgrounds
    for vbg_path in virtual_background_paths:
        vbg_name = vbg_path.stem
        is_blurred_bg = vbg_name.endswith("-blur")
        if is_blurred_bg:
            blurred_bg_name, _ = vbg_name.split("-")
            if blurred_bg_name != bg_name:
                continue
        create_vbg_masks(
            frames,
            mp_masks_path,
            vbg_path,
            target_resolution,
        )

    return True


def main(data_path, nprocs):
    bg_videos_path = data_path / "data" / "bg_videos"
    video_paths = list(bg_videos_path.glob("*.mp4"))
    with Pool(nprocs) as p:
        it = p.imap_unordered(
            ft.partial(process_video, data_path=data_path), video_paths
        )
        assert all(tqdm(it, total=len(video_paths)))


@click.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--nprocs",
    "-p",
    default=1,
    type=int,
)
def cli(data_path, nprocs):
    main(Path(data_path), nprocs)


if __name__ == "__main__":
    cli()
