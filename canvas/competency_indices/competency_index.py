import numpy as np
from .score_function import ScoreFunction

class CompetencyIndex:
    """
    Compute CI = R*/(err + R*), where:
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
        return self._to_ci(err, rs)

    # ---------------------------- helpers ----------------------------
    # Compute CI = R*/(err + R*)
    def _to_ci(self, err, r_star):
        if np.isscalar(err):
            rs = float(r_star) if np.isscalar(r_star) else float(np.asarray(r_star).ravel()[0])
            if not np.isfinite(rs) or rs <= 0:
                return float("nan")
            val = rs / (float(err) + rs)
            return float(min(val, 1.0)) if self.clip else float(val)

        e = np.asarray(err, dtype=np.float64)
        if e.size == 0:
            return e
        rs = self._broadcast_rstar(r_star, e.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = rs / (e + rs)
        ci[~np.isfinite(ci)] = np.nan
        if self.clip:
            ci = np.minimum(ci, 1.0)
        return ci
    
    @staticmethod
    def aci_lower_bound(U, r_star):
        """
        For ACI upper bound U(=confidence interval half-width),
        return ACI lower bound L = R*/(U + R*) (allow scalar/series)
        """
        if np.isscalar(U):
            rs = float(r_star) if np.isscalar(r_star) else float(np.asarray(r_star).ravel()[0])
            if not np.isfinite(rs) or rs <= 0:
                return float("nan")
            return float(min(rs/(float(U)+rs), 1.0))
        U = np.asarray(U, dtype=np.float64)
        rs = float(r_star) if np.isscalar(r_star) else float(np.asarray(r_star).ravel()[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = rs / (U + rs)
        ci[~np.isfinite(ci)] = np.nan
        return np.minimum(ci, 1.0)


