import numpy as np
from .score_function import ScoreFunction

class CompetencyIndex:
    """
    Compute CI = (R* - err) / R*, where:
      - R*: user-provided tolerance (scalar or per-step array)
      - err: score from ScoreFunction (series or scalar)
    """

    def __init__(self, case="traj", r_star=0.5, return_type="series", clip=True):
        # case: "traj" | "control" | "obj" | "ctrl_cost"
        # return_type: for "traj"/"control" → "series"|"mean"|"final"
        self.case = case
        self.r_star = r_star
        self.return_type = return_type
        self.clip = bool(clip)
        self._sf = ScoreFunction(case=case, return_type=return_type)

    def __call__(self, *, r_star=None, return_type=None, **kwargs):
        """
        Forward kwargs to ScoreFunction and convert the returned err to CI.
        You can override r_star/return_type per call.
        """
        rs = self.r_star if r_star is None else r_star
        rt = self.return_type if return_type is None else return_type

        # Get error from ScoreFunction (np.ndarray for series, float for scalars)
        err = self._sf(case=self.case, return_type=rt, **kwargs)

        # Compute CI
        return self._to_ci_2(err, rs)

    # ---------------------------- helpers ----------------------------
    # Compute CI = (R* - err) / R*
    def _to_ci(self, err, r_star):
        # Handle scalar vs array errors uniformly
        if np.isscalar(err):
            return self._ci_scalar(float(err), r_star)
        # err is array-like
        e = np.asarray(err, dtype=np.float64)
        if e.size == 0:
            return e  # empty series
        rs = self._broadcast_rstar(r_star, e.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = (rs - e) / rs
        ci[~np.isfinite(ci)] = np.nan
        if self.clip:
            ci = np.minimum(ci, 1.0)
        return ci
    
    # Compute CI = exp(- err / R*)
    def _to_ci_2(self, err, r_star):
        # Handle scalar vs array errors uniformly
        if np.isscalar(err):
            return self._ci_scalar_2(float(err), r_star)
        # err is array-like
        e = np.asarray(err, dtype=np.float64)
        if e.size == 0:
            return e  # empty series
        rs = self._broadcast_rstar(r_star, e.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = np.exp(-e/rs)
        ci[~np.isfinite(ci)] = np.nan
        if self.clip:
            ci = np.minimum(ci, 1.0)
        return ci

    def _ci_scalar(self, err, r_star): # for _to_ci
        # r_star can be scalar or array-like; if array, use its first element
        if np.isscalar(r_star):
            rs = float(r_star)
        else:
            rsa = np.asarray(r_star, dtype=np.float64).ravel()
            rs = float(rsa[0]) if rsa.size else np.nan
        if not np.isfinite(rs) or rs <= 0:
            return float("nan")
        ci = (rs - float(err)) / rs
        if self.clip:
            ci = min(ci, 1.0)
        return float(ci)
    
    def _ci_scalar_2(self, err, r_star): # for _to_ci_2
        # r_star can be scalar or array-like; if array, use its first element
        if np.isscalar(r_star):
            rs = float(r_star)
        else:
            rsa = np.asarray(r_star, dtype=np.float64).ravel()
            rs = float(rsa[0]) if rsa.size else np.nan
        if not np.isfinite(rs) or rs <= 0:
            return float("nan")
        ci = np.exp(- float(err)/rs)
        if self.clip:
            ci = min(ci, 1.0)
        return float(ci)

    @staticmethod
    def _broadcast_rstar(r_star, shape):
        if np.isscalar(r_star):
            return np.full(shape, float(r_star), dtype=np.float64)
        arr = np.asarray(r_star, dtype=np.float64)
        if arr.shape == shape:
            return arr
        try:
            return np.broadcast_to(arr, shape).astype(np.float64, copy=False)
        except ValueError:
            # Fallback: use first entry if broadcasting fails
            base = float(arr.ravel()[0]) if arr.size else np.nan
            return np.full(shape, base, dtype=np.float64)


