from .wrapper_predictor import BasePredictors 
from .koopcast import _load_cfg, _resolve_device, _load_K, _load_mdn, evaluate_from_obs_all
HISTORY_LENGTH_DEFAULT = 8
PRED_LENGTH_DEFAULT    = 12
MDN_COMPONENTS_DEFAULT = 6
MDN_HIDDEN_DEFAULT     = 128
SIGMA_MIN_DEFAULT      = 0.05
NEIGHBOR_RADIUS_DEFAULT = 5.0
class Koopcast_predictor(BasePredictors):
    """
    Single entrypoint:
      evaluator = MDNKoopmanEvaluator(cfg_path, mdn_pt_path=..., device='cuda:0')
      preds = evaluator(obs_dict)   # __call__ -> evaluate_from_obs_all
    """
    def __init__(
        self,
        prediction_len: int = 12,
        history_len: int = 8,
        dt=0.1,
        K_path: str = "src/canvas/predictors/koopcast/data/univ_koopman_K_1.npy",
        cfg_path: str="src/canvas/predictors/koopcast/data/univ_cfg.json",
        mdn_pt_path: str="src/canvas/predictors/koopcast/data/univ_mdn.pt",
        device: str ="cpu" ,
        mdn_K: int= MDN_COMPONENTS_DEFAULT,
        mdn_hidden: int= MDN_HIDDEN_DEFAULT,
        sigma_min: float =SIGMA_MIN_DEFAULT,
    ):
        super().__init__(
            prediction_len=prediction_len,
            history_len=history_len,
            dt=dt,
            device=device,
        )
        cfg = _load_cfg(cfg_path)
        self.H = int(cfg["H"])
        self.P = int(cfg["P"])
        self.max_neighbors = int(cfg["max_neighbors"])
        self.neighbor_radius = float(cfg["neighbor_radius"])
        self.neighbor_relative = bool(cfg["neighbor_relative"])

        self.device = _resolve_device(device or cfg.get("device", "cpu"))

        # Choose MDN weights path: explicit arg > cfg['mdn_path']
        mdn_path = mdn_pt_path or cfg.get("mdn_path", None)
        if mdn_path is None:
            raise ValueError("MDN path not provided. Pass mdn_pt_path=... or add 'mdn_path' to cfg JSON.")

        self.K = _load_K(cfg["K_path"] if K_path is None else K_path)  # [D_z,D_z]
        self.mdn = _load_mdn(
            mdn_path,
            in_dim=int(cfg["feature_in_dim"]),
            device=self.device,
            mdn_K=int(cfg.get("mdn_K", mdn_K)),
            mdn_hidden=int(cfg.get("mdn_hidden", mdn_hidden)),
            sigma_min=float(cfg.get("sigma_min", sigma_min)),
        )

    def __call__(self, obs: dict) -> dict:
        return evaluate_from_obs_all(
            obs=obs,
            H=self.H,
            P=self.P,
            mdn=self.mdn,
            K=self.K,
            device=self.device,
            max_neighbors=self.max_neighbors,
            neighbor_radius=self.neighbor_radius,
            neighbor_relative=self.neighbor_relative,
        )
