"""Preprocess behavior and raw EDF data into fixation scanpaths (Eyelink fixations).

Example::

    python -m salience_bias.analyses.preprocess.workflow.preprocess \\
        --subject SUBJECT --datacode YYYYMMDD_HHMMSS \\
        --dataset-name steer-search
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from salience_bias.analyses.preprocess.core.image_map import load_imgn_exp_to_data
from salience_bias.utils import psychopy
from salience_bias.utils.calibrate import Affine, consensus_calib, segment_trials
from salience_bias.utils.dataset import ROOT, extract_info, load_array

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, required=True)
parser.add_argument("--datacode", type=str, required=True)
parser.add_argument(
    "--dataset-name",
    type=str,
    default="steer-search",
    help="Raw/interim folder name and config/<name>.yaml stem (default: steer-search).",
)
parser.add_argument(
    "--calib-landmarks",
    type=int,
    nargs="+",
    default=[0, 25, 50, 75, 100],
)
parser.add_argument(
    "--consensus-thres",
    type=float,
    default=1.0,
    help="Consensus threshold for calibration",
)
parser.add_argument(
    "--consensus-min-valid-targets",
    type=int,
    default=9,
    help="Minimum number of valid targets for consensus calibration (among 17 targets)",
)
parser.add_argument(
    "--eyetracking",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Load EDF, calibrate, and emit fixation scanpaths (default: on; --no-eyetracking for behavior-only)",
)
args = parser.parse_args()

subject = args.subject
datacode = args.datacode

data_dir_name = args.dataset_name
base_dir = ROOT / "private" / "data" / "raw" / data_dir_name
interim_dir = ROOT / "private" / "data" / "interim" / data_dir_name
interim_dir.mkdir(parents=True, exist_ok=True)

calib_landmarks = dict(enumerate(args.calib_landmarks))
valid_calib_indices = list(calib_landmarks.keys())
valid_trial_indices = list(range(1, 101))

n_tail = 15
consensus_thres = args.consensus_thres
consensus_min_valid_targets = args.consensus_min_valid_targets

imgn_exp_to_data = load_imgn_exp_to_data(data_dir_name)

with open(base_dir / "meta.yaml", "r") as f:
    meta = yaml.safe_load(f)

meta = {
    sub: {
        fname: meta for fname, meta in sessions.items()
        if not meta.get("exclude", False) and meta.get("experiment") == "search"
    }
    for sub, sessions in meta.items()
}

if "anchor" in meta[subject][datacode]:
    anchor = True
else:
    anchor = False

with open(base_dir / subject / "behav" / f"{datacode}.json", "r") as f:
    data = json.load(f)

if anchor:
    if "fixation_positions" in data["data"]:
        anchor_positions = np.array(data["data"]["fixation_positions"]).T
    else:
        print("fixation_positions not found, using zeros...")
        anchor_positions = np.zeros([len(valid_trial_indices), 2])
else:
    anchor_positions = [None] * len(valid_trial_indices)

if args.eyetracking:
    import pyedfread

    samples, events, messages = pyedfread.read_edf(
        str(base_dir / subject / "eye" / f"{datacode}.edf")
    )

    calib_data = data["data"]["calibration_results"]
    print(f"found {len(calib_data)} calib data")

    calib_data = [calib_data[i] for i in valid_calib_indices]

    ses_indices = np.arange(len(calib_data))
    positions_target = load_array([c["positions_target"] for c in calib_data])
    positions_eye = load_array(
        [np.array(c["positions_eye"])[..., -n_tail:] for c in calib_data]
    )

    positions_eye, valid_ses_bool = consensus_calib(
        positions_target,
        positions_eye,
        consensus_thres=consensus_thres,
        consensus_min_valid_targets=consensus_min_valid_targets,
        return_valid_consensus=True,
    )
    print()

    if sum(valid_ses_bool) == 0:
        print("No valid calibration sessions found.")
        calib_params = {}
    else:
        print("valid_ses_indices", ses_indices[valid_ses_bool])
        print()

        calib_landmarks_valid = {
            i: calib_landmarks[i] for i in ses_indices[valid_ses_bool]
        }
        segments_valid = segment_trials(
            calib_landmarks_valid, all_trials=valid_trial_indices
        )

        calib_params = {}
        for i_ses, t_ses, e_ses in zip(
            ses_indices[valid_ses_bool],
            positions_target[valid_ses_bool],
            positions_eye[valid_ses_bool],
        ):
            print(
                f"valid session {i_ses} : removed data "
                f"{np.isnan(e_ses.mean(axis=(1, 2))).sum()}"
            )
            calib_param = Affine()
            calib_param.fit(t_ses, e_ses)
            calib_params[i_ses] = {
                "affine": calib_param,
                "trials": segments_valid[i_ses],
            }

    num_pattern = r"([-+]?\d*\.?\d+)"
    onset_info = {}
    for m in messages.message:

        if not any(tag in m for tag in ("FIXATION_ONSET", "IMAGE_ONSET", "RESPONSE")):
            continue

        trial_match = re.search(r"TRIAL[ =](\d+)", m)
        trial = int(trial_match.group(1)) if trial_match else None
        if trial is None:
            continue

        onset_info.setdefault(trial, {})

        if "FIXATION_ONSET" in m:
            onset_info[trial].update({
                "ET_fix": extract_info(m, rf"ET={num_pattern}", float),
                "PT_fix": extract_info(m, rf"PT={num_pattern}", float),
            })

        if "IMAGE_ONSET" in m:
            onset_info[trial].update({
                "ET_stim": extract_info(m, rf"ET={num_pattern}", float),
                "PT_stim": extract_info(m, rf"PT={num_pattern}", float),
                "pre_stim": extract_info(m, rf"PRE={num_pattern}", float),
                "flip_stim": extract_info(m, rf"FLIP={num_pattern}", float),
                "post_stim": extract_info(m, rf"POST={num_pattern}", float),
                "id_stim": extract_info(m, r"STIM_ID=(\d+)", int),
            })

        if "RESPONSE" in m:
            onset_info[trial].update({
                "ET_resp": extract_info(m, rf"ET={num_pattern}", float),
                "PT_resp": extract_info(m, rf"PT={num_pattern}", float),
                "key_resp": extract_info(m, r"KEY=([A-Za-z0-9_]+)"),
                "RT": extract_info(m, rf"RT={num_pattern}", float),
            })

    events_fix = events[events.type == "fixation"]
    t_stt = events_fix.start.values

    fixations_trials = {}
    for i_ses, v_ses in calib_params.items():

        affine = v_ses["affine"]
        trial_indices = v_ses["trials"]

        print("Affine matrix (A):\n", affine.A)
        print("Affine offset (b):", affine.b)

        for trial_idx in trial_indices:
            t_trial_stt = onset_info[trial_idx]["ET_stim"]
            t_trial_end = onset_info[trial_idx]["ET_resp"]

            fix_trial = events_fix[(t_stt > t_trial_stt) & (t_stt <= t_trial_end)]

            if len(fix_trial) == 0:
                gavx_trial = np.array([])
                gavy_trial = np.array([])
                onset_time = np.array([])
            else:
                gavx_trial, gavy_trial = psychopy.el2deg(
                    fix_trial.gavx.values, fix_trial.gavy.values
                )
                gav_trial = affine.transform(
                    np.stack([gavx_trial, gavy_trial], axis=-1)
                )
                gavx_trial, gavy_trial = gav_trial[:, 0], gav_trial[:, 1]
                onset_time = fix_trial.start.values - t_trial_stt

            fixations_trials[trial_idx] = {
                "x": gavx_trial,
                "y": gavy_trial,
                "t": onset_time,
            }

    scanpaths = []

    for tr_idx in sorted(fixations_trials.keys()):
        i = int(tr_idx) - 1
        tr_rt = data["data"]["rt"][i]
        tr_imgp = data["data"]["images"][i]
        tr_choice = data["data"]["response"][i]
        tr_anchor = anchor_positions[i]
        blknum = int(meta[subject][datacode]["block"])
        imgn = int(Path(tr_imgp).stem)
        tgtn = meta[subject][datacode]["target"]
        tr_onset = fixations_trials[tr_idx]["t"] / 1000.0
        tr_T = np.round(np.diff(np.concatenate([tr_onset, [tr_rt]])), 4)

        scanpath = {
            "block": blknum,
            "trial": int(tr_idx),
            "subject": subject,
            "image": imgn_exp_to_data[(blknum, imgn, tgtn)],
            "target": tgtn,
            "X": fixations_trials[tr_idx]["x"].tolist(),
            "Y": fixations_trials[tr_idx]["y"].tolist(),
            "T": tr_T.tolist(),
            "onset": tr_onset.tolist(),
            "length": len(tr_onset),
            "choice": {0: "present", 1: "absent"}[tr_choice],
            "RT": tr_rt,
        }

        if tr_anchor is not None:
            scanpath["anchor"] = tr_anchor.tolist()

        scanpaths.append(scanpath)

else:
    scanpaths = []

    for tr_idx_0, (tr_rt, tr_imgp, tr_choice) in enumerate(zip(
        data["data"]["rt"],
        data["data"]["images"],
        data["data"]["response"],
    )):
        blknum = int(meta[subject][datacode]["block"])
        imgn = int(Path(tr_imgp).stem)
        tgtn = meta[subject][datacode]["target"]
        tr_anchor = anchor_positions[tr_idx_0]

        scanpath = {
            "block": blknum,
            "trial": int(tr_idx_0 + 1),
            "subject": subject,
            "image": imgn_exp_to_data[(blknum, imgn, tgtn)],
            "target": tgtn,
            "choice": {0: "present", 1: "absent"}[tr_choice],
            "RT": tr_rt,
        }

        if tr_anchor is not None:
            scanpath["anchor"] = tr_anchor.tolist()

        scanpaths.append(scanpath)

(interim_dir / subject).mkdir(parents=True, exist_ok=True)
with open(interim_dir / subject / f"scanpaths_{datacode}.json", "w") as f:
    json.dump(scanpaths, f, indent=4)

print(f"saved scanpaths to {interim_dir / subject / f'scanpaths_{datacode}.json'}")
