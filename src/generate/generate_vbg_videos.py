""" Takes the videos with real surroundings as inputs and inserts a virtual background.
The necessary masks are created by `src/generate/generate_mp_masks.py` for MediaPipe.
For Zoom, it is necessary to pre-compute the masks using the AlphaDump mask dumping tool
for Zoom (not published). For the sample video zoom masks are available in data/zoom.
"""
import cv2
import ffmpeg
import click
import numpy as np
import functools as ft
from tempfile import TemporaryDirectory
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


def virtual_background(
    gs_mask_path,
    leak_masks_path,
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
            str(leak_masks_path / f"{i:06d}.png"),
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

def create_vbg_video(video_path, data_path, tmp_dir, bg_masks_path, leak_masks_path, tool):
    virtual_background_path = data_path / "virtual_backgrounds"
    virtual_background_paths = list(virtual_background_path.glob("*.png"))
    target_resolution = (1280, 720)
    name = video_path.stem
    number, gesture, bg = name.split("_")

    frames = decode_video(video_path)
    masks = read_masks(bg_masks_path / f"{number}_{gesture}_{bg}")

    for vbg_path in virtual_background_paths:
        with TemporaryDirectory(dir=tmp_dir) as tmp_directory:
            tmp_path = Path(tmp_directory)
            vbg_frames_path = tmp_path / "vbg_frames"
            vbg_frames_path.mkdir()
            vbg = vbg_path.stem
            is_blurred_bg = vbg.endswith("-blur")
            if is_blurred_bg:
                blurred_bg_name, _ = vbg.split("-")
                if blurred_bg_name != bg:
                    continue

            gs_mask_path = data_path / "data" / "greenscreen_masks" / f"{number}_{gesture}"
            leak_masks_video_path = leak_masks_path / f"{number}_{gesture}_{bg}" 
            leak_masks_video_path.mkdir(exist_ok=True)

            virtual_background(
                gs_mask_path,
                leak_masks_video_path,
                vbg_frames_path,
                frames.copy(),
                masks.copy(),
                vbg_path,
                target_resolution,
            )

            ffmpeg.input(
                str(vbg_frames_path / "%6d.png"),
                pattern_type="sequence",
                framerate=30,
            ).filter("fps", "30").filter("format", "yuv420p").output(
                str(
                    data_path
                    / "data"
                    / "vbg_videos"
                    / f"{number}_{gesture}_{bg}_{vbg}_{tool}.mp4"
                ),
                **{"q:v": 0},
            ).run(
                overwrite_output=True
            )

    return True


def process_video(video_path, tool, data_path, tmp_dir):
    if tool == "zoom":
        # Create zoom vbg videos
        bg_masks_path = data_path / "data" / "bg_zoom_masks"
        leak_masks_path = data_path / "data" / "zoom_leak_masks"
    elif tool == "mp":
        # Create mediapipe vbg videos
        bg_masks_path = data_path / "data" / "bg_mp_masks"
        leak_masks_path = data_path / "data" / "mp_leak_masks"
    else:
        # Invalid tool
        return False

    res = create_vbg_video(video_path, data_path, tmp_dir, bg_masks_path, leak_masks_path, tool)
    if not res:
        return False

    return True
    

def main(data_path, tool, tmp_dir, nprocs):
    bg_videos_path = data_path / "data" / "bg_videos"
    video_paths = list(bg_videos_path.glob("*.mp4"))
    with Pool(nprocs) as p:
        it = p.imap_unordered(
            ft.partial(process_video, data_path=data_path, tool=tool, tmp_dir=tmp_dir), video_paths
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
    "--tool",
    required=True,
    type=click.Choice(["zoom", "mp"]),
    help="The video conferenceing tool to create the virtual backgorund videos for.",
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
def cli(data_path, tool, tmp_dir, nprocs):
    main(Path(data_path), tool, tmp_dir, nprocs)


if __name__ == "__main__":
    cli()
