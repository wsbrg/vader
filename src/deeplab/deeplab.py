""" Create portrait masks using DeepLabV3.
"""
import cv2
import torch
import click
import os.path as osp
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


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


def batches(it, bs):
    while True:
        batch = []
        for i in range(0, bs):
            try:
                batch.append(next(it))
            except StopIteration:
                break
        if len(batch) == 0:
            break
        yield batch


def process_frame(frames, model):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_batch = torch.stack([preprocess(frame) for frame in frames])

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")

    with torch.no_grad():
        outputs = model(input_batch)["out"]
        outputs = outputs.argmax(1)
        mask = outputs == 15
        outputs[mask] = 255
        outputs[~mask] = 0

    return outputs.cpu()


def process_video(video_path, masks_path, model):
    frames = decode_video(video_path)
    batch_size = 8
    for batch_ctr, frames in enumerate(batches(frames, batch_size)):
        segmented_frames = process_frame(frames, model)
        for index, segmented in enumerate(segmented_frames):
            frame_ctr = batch_ctr * batch_size + index
            mask_path = masks_path / f"{frame_ctr:06d}.png"
            segmented = np.array(segmented)
            cv2.imwrite(str(mask_path), segmented)


def main(data_path, nprocs):
    if nprocs > 1:
        raise NotImplementedError("Parallel processing not implemented yet.")

    torch.hub.set_dir("/root/.cache/torch/hub")
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True
    )
    model.eval()

    if torch.cuda.is_available():
        model.to("cuda")
        print("[*] Cuda available")
    else:
        print("[*] Cuda not available")

    video_paths = list((data_path / "data" / "vbg_videos").glob("*.mp4"))
    for video_path in tqdm(video_paths):
        name, _ = osp.splitext(video_path.name)
        masks_path = data_path / "data" / "deeplab_masks" / name
        masks_path.mkdir(exist_ok=True)
        process_video(video_path, masks_path, model)


@click.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Data base directory path.",
)
@click.option(
    "--nprocs", "-p", type=int, default=1, help="Number of parallel processes."
)
def cli(data_path, nprocs):
    main(Path(data_path), nprocs)


if __name__ == "__main__":
    cli()
