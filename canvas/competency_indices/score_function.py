# Example score functions. User can customize or define a score function following the score function reference
import numpy as np

class ScoreFunction:
    """
    Score function interface for simulation:
      - case="traj":      trajectory prediction error (predicted vs GT future)
      - case="control":   controller output error ((v,w) difference per timestep)
      - case="obj":       error of the objectives (difference of optimal total cost(e.g. minimum total cost))
      - case="ctrl_cost": error of the control cost (difference of control-term cost)
      - case="custom":    user-defined score function, user can define the user-base score function here

    Return policy:
      - "traj":      returns series/mean/final depending on return_type
      - "control":   returns series/mean/final depending on return_type
      - "obj":       returns a scalar float (per-frame)
      - "ctrl_cost": returns a scalar float (per-frame)
      - "custom":    returns whatever your custom_fn returns

    Relative error:
      - If relative=True, errors for "obj" / "ctrl_cost" are computed as:
            abs(pred - gt) / (abs(gt) + eps)
        (Absolute numerator/denominator to keep the error nonnegative.)
    """

    def __init__(self, case="traj", return_type="series", xy_cols=(0, 1), vw_cols=(0, 1), custom_fn=None, relative=False, eps=1e-9):
        # Default settings
        self.case = case                # "traj" | "control" | "obj" | "ctrl_cost" | "custom"
        self.return_type = return_type  # "series" | "mean" | "final"
        self.xy_cols = tuple(xy_cols)   # columns for (x, y) in trajectories
        self.vw_cols = tuple(vw_cols)   # columns for (v, w) in controller plans
        self.custom_fn = custom_fn      # callable or None
        self.relative = bool(relative)  # relative error for obj/ctrl_cost
        self.eps = float(eps)           # denominator epsilon

    def __call__(self, case=None, return_type=None, **kwargs):
        c = case or self.case
        rtype = return_type or self.return_type

        if c == "traj":
            series = self._traj_series(
                prediction_res=kwargs["prediction_res"],
                gt_future=kwargs["gt_future"],
            )
            return self._reduce_series(series, rtype)

        if c == "control":
            series = self._control_series(
                ctrl_pred=kwargs["ctrl_pred"],
                ctrl_gt=kwargs["ctrl_gt"],
            )
            return self._reduce_series(series, rtype)  # "final" = last-step diff

        if c == "obj":
            J_pred = float(kwargs["minimal"])
            J_gt   = float(kwargs["minimal_gt"])
            return self._scalar_error(J_pred, J_gt)

        if c == "ctrl_cost":
            C_pred = float(kwargs["intermediate"]) + float(kwargs["terminal"]) + float(kwargs["control"])
            C_gt   = float(kwargs["intermediate_gt"]) + float(kwargs["terminal_gt"]) + float(kwargs["control_gt"])
            return self._scalar_error(C_pred, C_gt)

        if c == "custom":
            if self.custom_fn is None:
                raise NotImplementedError(
                    "No custom_fn provided. Pass custom_fn=... in constructor."
                )
            return self.custom_fn(**kwargs)

        raise ValueError("Unknown case: {}".format(c))

    # ------------------------- For trajectory metrics -------------------------
    def _traj_series(self, prediction_res, gt_future):
        """
        Per-step mean L2 between predicted and GT futures (PID-wise mean at each step).
        Uses all available steps (full horizon).
        """
        common = list(set(prediction_res.keys()) & set(gt_future.keys()))
        if not common:
            return np.array([], dtype=np.float64)

        per_pid_errs, max_T = [], 0
        for pid in common:
            p = np.asarray(prediction_res[pid], dtype=np.float64)
            g = np.asarray(gt_future[pid],      dtype=np.float64)

            # Validate shape and required columns
            if p.ndim < 2 or g.ndim < 2:
                continue
            if p.shape[1] <= max(self.xy_cols) or g.shape[1] <= max(self.xy_cols):
                continue

            # Overlapping horizon
            T = min(len(p), len(g))
            if T <= 0:
                continue

            p_xy = p[:T, list(self.xy_cols)]
            g_xy = g[:T, list(self.xy_cols)]

            # Skip non-finite rows
            if not (np.isfinite(p_xy).all() and np.isfinite(g_xy).all()):
                continue

            e = np.linalg.norm(p_xy - g_xy, axis=1)  # (T,)
            per_pid_errs.append(e)
            if T > max_T:
                max_T = T

        if not per_pid_errs or max_T == 0:
            return np.array([], dtype=np.float64)

        # NaN-pad to (num_pids, max_T) and step-wise nanmean
        M = len(per_pid_errs)
        pad = np.full((M, max_T), np.nan, dtype=np.float64)
        for i, e in enumerate(per_pid_errs):
            pad[i, :len(e)] = e

        return np.nanmean(pad, axis=0)  # (T_max,)

    # -------------------------- For controller-output metrics --------------------------
    def _control_series(self, ctrl_pred, ctrl_gt):
        """
        Per-step L2 difference between two (v,w) plans (pred vs GT).
        Uses the common length across the two sequences.
        """
        pred = np.asarray(list(ctrl_pred) if not hasattr(ctrl_pred, "shape") else ctrl_pred, dtype=np.float64)
        gt   = np.asarray(list(ctrl_gt)   if not hasattr(ctrl_gt, "shape")   else ctrl_gt,   dtype=np.float64)

        if pred.ndim != 2 or gt.ndim != 2:
            return np.array([], dtype=np.float64)
        if pred.shape[1] <= max(self.vw_cols) or gt.shape[1] <= max(self.vw_cols):
            return np.array([], dtype=np.float64)

        T = min(len(pred), len(gt))
        if T <= 0:
            return np.array([], dtype=np.float64)

        pred_vw = pred[:T, list(self.vw_cols)]
        gt_vw   = gt[:T,   list(self.vw_cols)]

        if not (np.isfinite(pred_vw).all() and np.isfinite(gt_vw).all()):
            return np.array([], dtype=np.float64)

        delta = pred_vw - gt_vw
        return np.linalg.norm(delta, axis=1)  # (T,)

    # --------------------------- Reduce helper ---------------------------
    @staticmethod
    def _reduce_series(series, rtype):
        """
        Convert a series to the requested return type:
          - "series": return as-is (np.ndarray)
          - "mean":   return float(np.nanmean(series)) or nan if empty
          - "final":  return float(series[-1]) or nan if empty
        """
        if rtype == "series":
            return series
        if series.size == 0:
            return float("nan")
        if rtype == "mean":
            return float(np.nanmean(series))
        if rtype == "final":
            return float(series[-1])
        raise ValueError("Unknown return_type: {}".format(rtype))

    # ------------------------- Scalar error helper ------------------------
    def _scalar_error(self, pred_value, gt_value):
        """
        Compute absolute or relative error between two scalars:
          - absolute: abs(pred - gt)
          - relative: abs(pred - gt) / (abs(gt) + eps)
        Returns a float.
        """
        diff = abs(float(pred_value) - float(gt_value))
        if not self.relative:
            return diff
        denom = abs(float(gt_value)) + self.eps
        return diff / denom

