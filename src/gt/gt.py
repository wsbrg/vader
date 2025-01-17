""" Create ground truths reconstructions based on the measured leakage.
"""
import click
import cv2
import colour
import numpy as np
from functools import partial
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


def read_masks(path):
    paths = sorted(path.glob("*.png"))
    for p in paths:
        yield cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def metric_CIEDE2000(image_1, image_2):
    image_1 = cv2.cvtColor(
        image_1.astype(np.float32) / 255, cv2.COLOR_BGR2Lab
    )
    image_2 = cv2.cvtColor(image_2.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    delta_e = colour.delta_E(image_1, image_2, method="CIE 2000")
    return delta_e


def process_video(leak_mask_path, data_path):
    is_zoom = "zoom_leak_masks" == leak_mask_path.parent.name
    name = leak_mask_path.stem
    number, gesture, bg_name = name.split("_")
    tool = "zoom" if is_zoom else "mp"
    output_name = f"{number}_{gesture}_{bg_name}_{tool}.png"
    output_path_gt = data_path / "data" / "ground_truths_masks" / output_name
    #outputpath_gt_deltae = data_path / "data" / "ground_truths_deltae" # TODO: REMOVE
    outputpath_gt_color = data_path / "data" / "ground_truths_color"
    vbgs_path = data_path / "virtual_backgrounds"
    bgs_path = data_path / "real_backgrounds"

    #outputpath_gt_deltae.mkdir(exist_ok=True, parents=True) # TODO: REMOVE
    outputpath_gt_color.mkdir(exist_ok=True, parents=True)

    gt_alpha = None
    for gt_frame_alpha in read_masks(leak_mask_path):
        if not isinstance(gt_alpha, np.ndarray):
            gt_alpha = np.zeros_like(gt_frame_alpha)
            #gt_color = np.zeros_like(frame)
        mask = gt_frame_alpha > gt_alpha
        gt_alpha[mask] = gt_frame_alpha[mask]
    cv2.imwrite(str(output_path_gt), gt_alpha)

    # Alpha blending
    gt_alpha = gt_alpha / 255
    mask = ~(gt_alpha > 0)
    gt_alpha = np.stack((gt_alpha,) * 3, axis=-1)
    color_mask = ~(gt_alpha > 0)

    bg = cv2.imread(str(bgs_path / f"{bg_name}.png"))
    bg = cv2.resize(bg, (1280, 720), cv2.INTER_LINEAR)
    for vbg_path in vbgs_path.glob("*.png"):
        vbg_name = vbg_path.stem
        vbg = cv2.imread(str(vbg_path))
        vbg = cv2.resize(vbg, (1280, 720), cv2.INTER_LINEAR)

        gt_color = np.multiply(vbg, 1 - gt_alpha) + np.multiply(bg, gt_alpha)
        gt_color[color_mask] = 0
        # gt_deltae = metric_CIEDE2000(gt_color, bg) TODO: REMOVE
        # gt_deltae[mask] = 255 # TODO: REMOVE
        fname = f"{number}_{gesture}_{bg_name}_{vbg_name}_{tool}.png"
        cv2.imwrite(str(outputpath_gt_color / fname), gt_color)
        #cv2.imwrite(str(outputpath_gt_deltae / fname), gt_deltae) # TODO: REMOVE

    return True


def main(data_path, nprocs):
    leak_mask_paths = list((data_path / "data" / "mp_leak_masks").iterdir())
    leak_mask_paths += list((data_path / "data" / "zoom_leak_masks").iterdir())

    with Pool(nprocs) as p:
        it = p.imap_unordered(partial(process_video, data_path=data_path), leak_mask_paths)
        assert all(tqdm(it, total=len(leak_mask_paths)))


@click.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Data path.",
)
@click.option(
    "--nprocs",
    "-p",
    default=1,
    type=int,
    help="Number of parallel processes.",
)
def cli(data_path, nprocs):
    main(Path(data_path), nprocs)


if __name__ == "__main__":
    cli()
