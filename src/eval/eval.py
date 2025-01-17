""" Evaluate the quality of the reconstructions based on the color distance to the
best reconstruction that would have been possible based on the measured leakage.
"""
import cv2
import click
import numpy as np
import os.path as osp
from multiprocessing import Pool
from pathlib import Path


GESTURES = ["standup", "wavearm", "wavehand", "tilthead", "turnhead", "interview"]
TOOLS = ["mp", "zoom"]
ATTACKS = ["sabra", "hilgefort", "vader"]


def eval_deltae_weighted(deltae_path):
    name, _ = osp.splitext(deltae_path.name)
    number, gesture, bg, vbg, tool = name.split("_")
    gt_path = (
        deltae_path.parent.parent.parent
        # Weights for weighted evaluation
        / "ground_truths_masks"
        / f"{number}_{gesture}_{bg}_{tool}.png" # For normal ground truth
    )

    # Read images
    delta_e = cv2.imread(str(deltae_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    delta_e = delta_e.flatten()
    assert np.max(gt) == 255 or np.max(gt) == 0
    gt = gt.flatten() / 255.0 # For weighted ground truth
    assert min(gt) == 0

    # Reconstruction performance score
    reconstructed = sum(gt[delta_e < 4])
    leaking = sum(gt)
    reconstruction_score = reconstructed / leaking

    return reconstructed, leaking, reconstruction_score


def evaluate(data_path, gesture, tool, attack, vbg, bg, nprocs):
    deltae_paths = (data_path / "data" / "deltae" / attack).glob("*.png")
    deltae_paths = filter(lambda path: f"_{gesture}_" in path.name, deltae_paths)
    deltae_paths = filter(lambda path: f"_{tool}." in path.name, deltae_paths)
    if vbg != None:
        deltae_paths = filter(lambda path: f"_{vbg}_" in path.name, deltae_paths)
    if bg != None:
        deltae_paths = filter(lambda path: f"_{bg}_" in path.name, deltae_paths)
    deltae_paths = list(deltae_paths)

    with Pool(nprocs) as p:
        reconstruction_scores = p.map(eval_deltae_weighted, deltae_paths)
        reconstruction_scores = list(zip(deltae_paths, reconstruction_scores))

    return reconstruction_scores


def main(data_path, nprocs, gesture, tool, attack):
    gesture_lookup = {
        "standup": "g",
        "wavearm": "a",
        "wavehand": "w",
        "tilthead": "n",
        "turnhead": "h",
        "interview": "i",
    }
    assert set(gesture_lookup.keys()) == set(GESTURES)

    output_path = data_path / "evaluation" / "raw"
    output_path.mkdir(parents=True, exist_ok=True)

    # Tools
    tools = [tool] if tool != None else TOOLS

    # Gestures
    gestures = (
        [(gesture, gesture_lookup[gesture])]
        if gesture != None
        else list(gesture_lookup.items())
    )

    # Attacks
    attacks = ATTACKS if attack == None else [attack]

    for attack in attacks:
        deltae_path = data_path / "data" / "deltae" / attack

        if not deltae_path.exists():
            continue

        # Virtual backgrounds and backgrounds
        bgs = []
        vbgs = []
        for fpath in deltae_path.iterdir():
            name, _ = osp.splitext(fpath.name)
            idx, _, bg, vbg, _ = name.split("_")
            if not bg in bgs:
                bgs.append(bg)
            if not vbg in vbgs:
                vbgs.append(vbg)

        print("Attack: ", attack)

        for tool in tools:
            for gesture, abbr in gestures:
                gesture_reconstruction_scores = []

                for vbg in vbgs:
                    vbg_mean_reconstruction_scores = []

                    for bg in bgs:
                        bg_reconstruction_scores = evaluate(data_path, abbr, tool, attack, vbg, bg, nprocs)

                        if len(bg_reconstruction_scores) == 0:
                            continue

                        gesture_reconstruction_scores += bg_reconstruction_scores
                        bg_reconstruction_scores = [score for _, (reconstructed, leaking, score) in bg_reconstruction_scores]
                        mean = np.mean(bg_reconstruction_scores)
                        vbg_mean_reconstruction_scores.append(mean)

                output_fname = f"{attack}_{gesture}_{tool}.dat"
                output_fpath = output_path / output_fname
                output_content = "\n".join(f"{deltae_path.name} {reconstructed} {leaking} {score:.4f}" for deltae_path, (reconstructed, leaking, score) in gesture_reconstruction_scores)
                output_fpath.write_text(output_content)

                gesture_reconstruction_scores = [score for _, (reconstructed, leaking, score) in gesture_reconstruction_scores]
                print(
                    f"{attack}-{tool}-{gesture}: mean {np.mean(gesture_reconstruction_scores)}, median {np.median(gesture_reconstruction_scores)}"
                )



@click.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    help="Data path.",
)
@click.option(
    "--nprocs",
    "-p",
    type=int,
    default=1,
    help="Number of parallel processes.",
)
@click.option(
    "--gesture",
    "-g",
    type=click.Choice(GESTURES),
    default=None,
    help="The gesture to evaluate (every gesture is evaluated if nothing is specified)",
)
@click.option(
    "--tool",
    "-t",
    type=click.Choice(TOOLS),
    default=None,
    help="The tool to evaluate (every tool is evaluated if nothing is specified)",
)
@click.option(
    "--attack",
    "-a",
    type=click.Choice(ATTACKS),
    default=None,
    help="The attack to evaluate (every attack is evaluated if nothing is specified)",
)
def cli(data_path, nprocs, gesture, tool, attack):
    main(Path(data_path), nprocs, gesture, tool, attack)


if __name__ == "__main__":
    cli()
