import time
import numpy as np
import os
import sys
import cv2
_DATA_DIR = os.path.dirname(__file__)
sys.path.append(_DATA_DIR)
from collections import defaultdict
from canvas import Predictors
from canvas.conformal_predictors.aci import DelayedACI
from canvas.controllers.controller import controllers
from canvas.conformal_predictors.scores import ActionDivergenceScoreFunction, PlanningRegretScoreFunction, PositionalDisplacementScoreFunction
from canvas.competency_indices.core import CompetencyIndex

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""
class Simulation():
    def __init__(self, environment, predictor, controller, goal, persistent_static_boxes, dataset, 
                 prediction_len, history_len, dt, t_begin, t_end, save_video=True, ci_mode: str="act", **kwargs):
        self.env = environment
        self.predictor = predictor
        self.controller = controllers(chosen_controller=controller, prediction_len=prediction_len, dt=dt, goal=goal)
        self.goal = goal
        self.persistent_static_boxes = persistent_static_boxes
        self.dataset_obj = dataset
        self.dataset_name = dataset.name
        self.prediction_len = prediction_len
        self.history_len = history_len
        self.dt = dt
        self.t_begin = t_begin
        self.t_end = t_end
        self.save_video = save_video
        # CI machinery
        self.ci_mode = ci_mode # 'pos'(PD) | 'act'(AD) | 'reg'(PR)
        # ci buffer
        self._ci_series = []
        self._ci_sum = 0.0      
        self._ci_count = 0 

        self.success_count = 0
        self.buffer_pos_x_result = []
        self.buffer_pos_y_result = []

    def set_buffer(self):
        self.success_count = 0
        self.buffer_pos_x_result = []
        self.buffer_pos_y_result = []

    def overall_frame_mean_ci(self) -> float:
        return (self._ci_sum / self._ci_count) if self._ci_count else float('nan')

    def run(self, times: int):
        import builtins as _b
        print = _b.print if self.verbose else (lambda *args, **kwargs: None)
        
        print("==================================")
        print("SIMULATION PIPELINE Started")
        print("==================================")
        frame = 0
        frames = []
        buffer_pos_x = []  # per-frame x within this run
        buffer_pos_y = []

        # ---------- Choose predictor ---------
        obj_predictor = Predictors(
            chosen_predictor=self.predictor,
            prediction_len=self.prediction_len,
            history_len=self.history_len,
            dt=self.dt,
            dataset=self.dataset_name,
            device='cpu'
        )
        obj_predictor_gt = Predictors(
            chosen_predictor="linear",
            prediction_len=self.prediction_len,
            history_len=self.history_len,
            dt=self.dt,
            dataset=self.dataset_name,
            device='cpu'
        )# linear predictor as comparison

        # ---------- CP module (update once per frame) : DelayedACI / CI configuration ---------
        indices = CompetencyIndex(prefix_len=self.t_begin)
        if self.ci_mode == "act":
            max_score_ad = (1.6 ** 2 + 1.4 ** 2) ** .5  # diameter of the action space
            score_ftn = ActionDivergenceScoreFunction(prediction_len=self.prediction_len)
            cp_module = DelayedACI(
                target_miscoverage_level=0.8,
                step_size=0.05,
                delay=self.prediction_len,
                max_score=max_score_ad,
                sample_size=20
            )
            indices.register(score_ftn, cp_module, name='action_divergence')
        elif self.ci_mode == "reg":
            max_score_pr = 800.
            score_ftn = PlanningRegretScoreFunction(prediction_len=self.prediction_len)
            cp_module = DelayedACI(
                target_miscoverage_level=0.8,
                step_size=0.05,
                delay=self.prediction_len,
                max_score=max_score_pr,
                sample_size=20
            )
            indices.register(score_ftn, cp_module, name='planning_regret')
        else:
            score_ftn = PositionalDisplacementScoreFunction(prediction_len=self.prediction_len, step=6)
        
        obs, _ = self.env.reset()
        truncated = False
        ego = obs['ego']  # ego : ego-vehicle(or robot)
        position_x, position_y, orientation_z = ego['position_x'], ego['position_y'], ego['orientation_z']
        cmd_linear_x, cmd_angular_z = 0.0, 0.0   # last cmd
        self._ci_series.clear()

        self.set_buffer()       

        while not truncated:
            detect_time = time.time()
            print(frame, position_x, position_y, orientation_z, cmd_linear_x, cmd_angular_z, time.time() - detect_time)

            # record robot trajectory
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # --------- Predictor (once per frame) ---------
            prediction_res = obj_predictor(obs['non-ego'])
            prediction_res_gt = obj_predictor_gt(obs['non-ego'])

            indices.update(obs)

            # ACI -> competency idx computation
            if frame >= self.prediction_len:
                indices.forward()
            else:
                indices.pad(0.5)

            # --------- Controller (once per frame, with predictions) ---------
            velocity, controller_info, minimum, intermediate, terminal, control, minimal = self.controller(
                    **obs['ego'],
                    linear_x=cmd_linear_x,
                    cmd_angular_z=cmd_angular_z,
                    boxes=self.persistent_static_boxes,
                    predictions=prediction_res,
                    confidence_intervals=confidence_intervals,
                    goal=self.goal
            )
            # For GT(Oracle based) : no status update here, just for get controller input for GT
            # TODO: implement the lagged ACI; this is cheating
            
            # --------- Feasibility handling ---------
            if not controller_info['feasible']:
                cmd_linear_x, cmd_angular_z = 0., 0.
                print(frame, 'No safe paths found, stopping robot movement for this frame.',
                      position_x, position_y, cmd_linear_x, cmd_angular_z, time.time() - detect_time)
            else:
                cmd_linear_x, cmd_angular_z = velocity[0]

            # --- Ground-truth future for energy (same frame) ---
            gt_future = self.dataset_obj.get_future(
                timestep=self.env.timestep,
                future_length=self.prediction_len,
                history_length=self.history_len,
            )

            # --------- Apply first control step ---------
            obs, terminated, truncated, simulation_info = self.env.step(np.array([cmd_linear_x, cmd_angular_z]))
            ego = obs['ego']
            position_x, position_y, orientation_z = ego['position_x'], ego['position_y'], ego['orientation_z']

            # --- Energy via score function (Eq. (2)/(3)/(4)) ---
            #   - 'traj' : no need controller component
            #   - 'control'/'obj' : need controller interface
            if self.ci_mode == "traj":
                E_t = self.sf(x=None, y_future=gt_future, yhat_future=prediction_res)
                eps = 1e-12
                if obj_predictor_gt is None:
                    raise RuntimeError("pairwise CI needs a baseline predictor (e.g., linear)")
                baseline_pred = prediction_res_gt
                E_base_t = self.sf(
                    x=None,
                    y_future=gt_future,
                    yhat_future=baseline_pred,
                    controller=None
                )
            else:
                # 'control'/'obj' : need controller API
                # action: controller(x, predictions=...) -> action(ndarray)
                # regret: controller.solve_with_cost(x, predictions=...) -> (u, J_scalar)
                # If the current does not provide the controller API, use 'traj' first
                E_t = self.sf(x=obs['ego'], y_future=gt_future, yhat_future=prediction_res, controller=self.controller)
                eps = 1e-12
                if obj_predictor_gt is None:
                    raise RuntimeError("pairwise CI needs a baseline predictor (e.g., linear)")
                baseline_pred = prediction_res_gt
                E_base_t = self.sf(
                    x=obs['ego'],
                    y_future=gt_future,
                    yhat_future=baseline_pred,
                    controller=self.controller
                )
            I_t = (E_base_t + eps) / (E_t + E_base_t + 2*eps)
            r_t = (E_t + eps) / (E_base_t + eps)

            # --- ACI upper bound for energy, then lower CI bound (paper) ---
            U_t = self.aci_energy.update(score=r_t)
            L_t = 1.0 / (1.0 + U_t)

            print("CI_lower(L_t): ", L_t)
            print("CI(I_t): ", I_t)

            if np.isfinite(L_t):
                self._ci_sum += L_t
                self._ci_count += 1

            # --------- Rendering the situation ----------
            if self.verbose:
                self._ci_series.append(L_t)
                fig, ax = self.env.render(c=self._ci_series + [float(L_t)])

                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())  # shape : (h, w, 4), RGBA
                rgb  = rgba[:, :, :3].copy()                 # drop alpha
                frames.append(rgb)
                fig.savefig(os.path.join('./viz_example', '{:03d}.png'.format(self.env.timestep)), bbox_inches='tight', pad_inches=0)

            # --------- Goal check ---------
            if terminated or self.env.timestep >= self.dataset_obj.max_timesteps - 1:
                print(frame, 'Goal reached!')
                self.success_count += 1
                break

            frame += 1

        if self.verbose:
            if frames:
                # Ensure even dims for H.264
                H, W = frames[0].shape[:2]
                H2, W2 = H - (H % 2), W - (W % 2)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = os.path.join('./viz_example', "trajectory.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, 1.0/self.env.dt, (W2, H2))

                for fr in frames:
                    if fr.shape[0] != H2 or fr.shape[1] != W2:
                        fr = fr[:H2, :W2]
                    writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                writer.release()






