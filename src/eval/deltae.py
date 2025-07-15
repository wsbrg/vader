""" Measure the CIEDE2000 color distance between a reconstructed image and the
best possible reconstruction that would have been possible based on the measured
leakage.
"""
import cv2, random
import numpy as np
import click
import colour
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from functools import partial


MAX_CHUNKSIZE = 128


def metric_CIEDE2000(reconstruction, background, gt_mask):
    gt_mask = gt_mask > 0
    reconstruction = cv2.cvtColor(
        reconstruction.astype(np.float32) / 255, cv2.COLOR_BGR2Lab
    )
    background = cv2.cvtColor(background.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    delta_e = colour.delta_E(reconstruction, background, method="CIE 2000")
    delta_e[~gt_mask] = 254.0

    return delta_e


def process_image(image_path, data_path):
    attack = image_path.parent.name
    name = image_path.stem
    number, gesture, bg, vbg, tool = name.split("_")
    gt_mask_path = (
        data_path / "data" / "ground_truths_masks" / f"{number}_{gesture}_{bg}_{tool}.png"
    )

    bg_path = data_path / "data" / "ground_truths_color" / f"{name}.png"
    output_dir_path = data_path / "data" / "deltae" / attack
    output_dir_path.mkdir(exist_ok=True)
    output_path = output_dir_path / f"{name}.png"

    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(str(bg_path))
    reconstruction = cv2.imread(str(image_path))

    background = cv2.resize(
        background,
        (gt_mask.shape[1], gt_mask.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    ciede20000_metric = metric_CIEDE2000(reconstruction, background, gt_mask)

    # Save ciede2000 data as image
    cv2.imwrite(str(output_path), ciede20000_metric)

    return True


def process_images(data_path, nprocs, seed):
    image_paths = list((data_path / "data" / "reconstructed").glob("*/*.png"))
    random.seed(seed)
    random.shuffle(image_paths)
    chunksize = min(max(len(image_paths) // nprocs, 1), MAX_CHUNKSIZE)

    with Pool(nprocs) as p:
        results = p.imap_unordered(
            partial(process_image, data_path=data_path),
            image_paths,
            chunksize=chunksize,
        )
        assert all(tqdm(results, total=len(image_paths)))


@click.command()
@click.option(
    "--data-path",
    "-d",
    type=click.Path(file_okay=False, dir_okay=True, exists=True),
    required=True,
    help="Data directory.",
)
@click.option(
    "--nprocs",
    "-p",
    type=int,
    default=1,
    help="Number of parallel processes.",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=42,
    help="Random seed.",
)
def cli(data_path, nprocs, seed):
    process_images(Path(data_path), nprocs, seed)


if __name__ == "__main__":
    cli()
