"""Offline calibration routines for eye-tracking data.

This module is the canonical implementation (preferred name: `calibrate`).
"""

import numpy as np

# ============================================================================
# eye calibration functions
# ============================================================================
class Affine:
    """class for affine transformation"""

    def __init__(self):
        # Initialize A and b as None, to be set after fitting
        self.A = None
        self.b = None

    def fit(self, xy_targets, xy_eyes):
        """
        Fits the affine transformation matrix and translation vector.

        Parameters:
        xy_targets (numpy array): [N, 2] matrix representing the target coordinates.
        xy_eyes (numpy array): [N, 2] matrix representing the eye-tracking coordinates.

        Sets:
        self.A : 2x2 affine transformation matrix.
        self.b : 2-dimensional translation vector.
        """
        if xy_eyes.ndim == 3:
            N, _, n_repeat = xy_eyes.shape
            xy_eyes = xy_eyes.transpose((0, 2, 1)).reshape(N * n_repeat, 2)
            xy_targets = np.repeat(xy_targets, n_repeat, axis=0)

        mask = ~np.isnan(xy_eyes).any(axis=1)
        xy_eyes = xy_eyes[mask]
        xy_targets = xy_targets[mask]

        # Add a column of ones for the translation term
        xy_eyes_augmented = np.hstack([xy_eyes, np.ones((xy_eyes.shape[0], 1))])

        # Solve for the affine transformation using least squares
        params, _, _, _ = np.linalg.lstsq(xy_eyes_augmented, xy_targets, rcond=None)

        self.A = params[:2, :].T  # 2x2 affine matrix
        self.b = params[2, :]  # 2x1 translation vector

    def transform(self, xy_eyes):
        """
        Applies the fitted affine transformation to the eye-tracking coordinates.

        Parameters:
        xy_eyes (numpy array): [N, 2] matrix representing the eye-tracking coordinates.
            or [N, 2, n_repeat] matrix representing the eye-tracking coordinates with n_repeat repeats.

        Returns:
        xy_transformed (numpy array): [N, 2] matrix representing the transformed coordinates.
        """

        if self.A is None or self.b is None:
            raise ValueError("The affine transformation has not been fitted yet.")

        # Handle both 2D and 3D cases
        is_3d = xy_eyes.ndim == 3
        if is_3d:
            N, _, n_repeat = xy_eyes.shape
            xy_eyes = xy_eyes.transpose((0, 2, 1)).reshape(N * n_repeat, 2)

        # Apply the transformation: XY_transformed = XY_eye @ A.T + T
        xy_transformed = xy_eyes @ self.A.T + self.b

        # Reshape back to original shape if it was 3D
        if is_3d:
            return xy_transformed.reshape(N, n_repeat, 2).transpose((0, 2, 1))
        return xy_transformed


def eval_rmse(
    fit_xy_targets,
    fit_xy_eyes,
    fit_affine=None,
    transform_xy_targets=None,
    transform_xy_eyes=None,
):
    """returns evaluation RMSE vector"""
    if transform_xy_targets is None:
        transform_xy_targets = fit_xy_targets
    if transform_xy_eyes is None:
        transform_xy_eyes = fit_xy_eyes

    if fit_affine is None:
        fit_affine = Affine()
        fit_affine.fit(fit_xy_targets, fit_xy_eyes)

    targ = transform_xy_targets
    trans = fit_affine.transform(transform_xy_eyes)
    rmse = np.sum((trans - targ[:, :, None]) ** 2, axis=1)
    rmse = np.sqrt(np.mean(rmse, axis=-1))
    return rmse


def consensus_calib(
    xy_targets,
    xy_eyes,
    baseline_calib=True,
    baseline_thres=2.0,
    baseline_max_iter=5,
    baseline_affine=None,
    consensus_thres=1.0,
    consensus_max_iter=5,
    consensus_min_valid_targets=None,
    verbose=True,
    return_valid_consensus=False,
):
    """
    [1] (baseline session) computes the best worst-case RMSE session without affine transform, then construct baseline_affine from the best ses.
    [2] (baseline calibration) computes checks (otherwise NaN) based on baseline_affine
    [3] (consensus calibration) computes checks (otherwise NaN) based on individual sessions

    inputs
    ------
        xy_targets [n_ses][N_target, 2] representing the target coordinates.
        xy_eyes [n_ses][N_target, 2, ...] representing the eye-tracking coordinates. (ususally ... = 31, n_samples)

        baseline_calib : whether to perform a baseline calibration
        baseline_thres : threshold for the baseline calibration
        baseline_max_iter : maximum iteration for the baseline calibration
        baseline_affine : affine object for the "None" baseline calibration

        consensus_thres : threshold for the consensus calibration
        consensus_max_iter : maximum iteration for the consensus calibration
        consensus_min_valid_targets : if not None, require this many targets to remain valid (non-NaN) for a session to be marked valid

        return_valid_consensus : if True, return a boolean list of which sessions converged before the max iteration limit

    outputs
    -------
        xy_eyes : updated xy_eyes
    """
    n_ses = len(xy_targets)

    if baseline_calib and baseline_affine is None:
        if verbose:
            print("Baseline calibration for prelim diagnosis...\n")
        worst_rmse = []
        for s in range(n_ses):
            err = eval_rmse(xy_targets[s], xy_eyes[s])
            worst_rmse.append(np.nanmax(err) if np.any(np.isfinite(err)) else np.nan)
        worst_rmse = np.array(worst_rmse)
        if not np.any(np.isfinite(worst_rmse)):
            raise ValueError("No session has valid eye data for baseline calibration.")
        base_ses = int(np.nanargmin(worst_rmse))

        if verbose:
            print("Worst-case RMSE for each session in RMSE")
            print(worst_rmse.tolist())
            print(f"Using session : {base_ses} as a baseline...\n")

        baseline_affine = Affine()
        baseline_affine.fit(xy_targets[base_ses], xy_eyes[base_ses])

    if baseline_calib:
        if verbose:
            print("\nStarting baseline calibration...")
        for ses in range(n_ses):
            for _ in range(baseline_max_iter):
                error = eval_rmse(None, None, baseline_affine, xy_targets[ses], xy_eyes[ses])
                worst_idx = np.argsort(-error)[0]
                worst_val = error[worst_idx]

                if worst_val > baseline_thres:
                    if verbose:
                        print(
                            f"Session {ses}, worst RMSE at position {worst_idx} is {worst_val} > {baseline_thres}"
                        )
                        print("...rejecting the data")
                    xy_eyes[ses][worst_idx] = np.nan

                else:
                    if verbose and np.isfinite(worst_val):
                        print(
                            f"Session {ses}, worst RMSE at position {worst_idx} is {worst_val} < {baseline_thres}"
                        )
                        print("...converged")
                    break

    if verbose:
        print("\nStarting consensus calibration...")
    ses_indices_valid_consensus = np.zeros(n_ses, dtype=bool)

    for ses in range(n_ses):
        for _ in range(consensus_max_iter):
            error = eval_rmse(xy_targets[ses], xy_eyes[ses])
            worst_idx = np.argsort(-error)[0]
            worst_val = error[worst_idx]

            if worst_val > consensus_thres:
                if verbose:
                    print(
                        f"Session {ses}, worst RMSE at position {worst_idx} is {worst_val} > {consensus_thres}"
                    )
                    print("...rejecting the data")
                xy_eyes[ses][worst_idx] = np.nan

            else:
                if np.isfinite(worst_val):
                    n_valid = (
                        np.isfinite(xy_eyes[ses])
                        .any(axis=tuple(range(1, xy_eyes[ses].ndim)))
                        .sum()
                    )
                    if consensus_min_valid_targets is None or n_valid >= consensus_min_valid_targets:
                        ses_indices_valid_consensus[ses] = True
                        if verbose:
                            print(
                                f"Session {ses}, worst RMSE at position {worst_idx} is {worst_val} < {consensus_thres}"
                            )
                            print("...converged")
                    elif verbose:
                        print(
                            f"Session {ses}, converged but only {n_valid} valid targets < {consensus_min_valid_targets}"
                        )
                break

    if return_valid_consensus:
        return xy_eyes, ses_indices_valid_consensus

    return xy_eyes


def segment_trials(landmark_trials, n_trials=None, all_trials=None):
    """segment trials based on membership to the closest landmark.

    inputs
    ------
    landmark_trials : list/array of landmark trial numbers or
        dict of {session_index: landmark_trial_number}
        the order of landmarks determines segment order (or key association).
    n_trials : int, optional
        If `all_trials` is None, create trials as np.arange(n_trials).
        If None, inferred from the last two landmarks (same as original behavior).
    all_trials : array-like, optional
        Explicit trial numbers to segment (e.g., [2, 5, 7, 10, ...]).
        If provided, `n_trials` is ignored.

    returns
    -------
    segments : list of np.ndarray (trial numbers), if landmark_trials is list/array
        segments[i] are the trial numbers whose closest landmark is landmark_trials[i].
    segments : dict {session_index: np.ndarray (trial numbers)}, if landmark_trials is dict
        Each key maps to the trials closest to that landmark trial number.
    """
    # preserve dict keys if provided
    is_dict = isinstance(landmark_trials, dict)
    if is_dict:
        keys = list(landmark_trials.keys())
        landmark_vals = np.array([landmark_trials[k] for k in keys])
        landmark_arr = landmark_vals
    else:
        keys = None
        landmark_arr = np.array(landmark_trials)

    # build all_trials
    if all_trials is None:
        if n_trials is None:
            # original heuristic; assumes at least 2 landmarks
            n_trials = landmark_arr[-1] + (landmark_arr[-1] - landmark_arr[-2])
        all_trials_arr = np.arange(n_trials)
    else:
        all_trials_arr = np.array(all_trials)

    # compute closest landmark for each trial number
    dists = np.abs(all_trials_arr[:, None] - landmark_arr[None, :])
    closest_landmark = np.argmin(dists, axis=1)

    # group by closest landmark, returning *trial numbers* (not indices)
    seg_list = [all_trials_arr[closest_landmark == i] for i in range(len(landmark_arr))]

    if is_dict:
        return {k: seg_list[i] for i, k in enumerate(keys)}
    return seg_list

