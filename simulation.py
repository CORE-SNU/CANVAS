import time
import numpy as np
import os
import sys
import cv2
_DATA_DIR = os.path.dirname(__file__)
sys.path.append(_DATA_DIR)
from collections import defaultdict
from canvas import Predictors
from canvas.competency_indices.score_function import ScoreFunction
from canvas.competency_indices.aci import AdaptiveEnergyCI

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""
class Simulation():
    def __init__(self, environment, predictor, controller, cp_module, goal, max_pedestrian, persistent_static_boxes, dataset, 
                 prediction_len, history_len, dt, save_video=True, r_star: float=0.5, ci_mode: str="traj", **kwargs):
        self.env = environment
        self.predictor = predictor
        self.controller = controller
        self.cp_module = cp_module
        self.goal = goal
        self.max_ped = max_pedestrian
        self.persistent_static_boxes = persistent_static_boxes
        self.dataset_obj = dataset
        self.dataset_name = dataset.name
        self.prediction_len = prediction_len
        self.history_len = history_len
        self.dt = dt
        self.save_video = save_video
        # CI machinery
        self.r_star = float(r_star)
        self.ci_mode = ci_mode # 'traj' | 'control' | 'obj'
        # Horizon aggregation: traj/control ->'max', obj -> 'mean'
        hagg = "max" if ci_mode in ("traj", "control") else "mean"
        mode = {"traj": "pos", "control": "action", "obj": "regret"}[ci_mode]

        # Section IV score function (Eq. (2)(3)(4))
        self.sf = ScoreFunction(mode=mode, horizon_agg=hagg)

        # ACI for energy stream (Section IV: U_t upper bound)
        self.aci_energy = AdaptiveEnergyCI(alpha=0.1, step_size=0.05)

        # ci buffer
        self._ci_series = []

        self.ci_boot = 0
        self.ci_ref_q = 0.5
        self.E_ref_fixed = None
        self._E_boot = []

        self.buffer_collision_rate = []
        self.buffer_infeasible_rate = []
        self.buffer_avg_minimal_cost = []
        self.buffer_avg_intermediate_cost = []
        self.buffer_avg_terminal_cost = []
        self.buffer_avg_control_cost = []
        self.buffer_prediction_times = []
        self.buffer_travel_times = []
        self.success_count = 0
        self.buffer_pos_x_result = []
        self.buffer_pos_y_result = []

    def set_buffer(self):
        self.buffer_collision_rate = []
        self.buffer_infeasible_rate = []
        self.buffer_avg_minimal_cost = []
        self.buffer_avg_intermediate_cost = []
        self.buffer_avg_terminal_cost = []
        self.buffer_avg_control_cost = []
        self.buffer_prediction_times = []
        self.buffer_travel_times = []
        self.success_count = 0
        self.buffer_pos_x_result = []
        self.buffer_pos_y_result = []

    def run(self, times: int):
        print("==================================")
        print("SIMULATION PIPELINE Started")
        print("==================================")
        frame = 0
        frames = []
        infeasible_count = 0
        infeasible_streak = 0
        max_infeasible_streak = 10
        collision_count = 0
        is_success = False

        buffer_infeasibility = []
        minimum_cost = []
        buffer_pos_x = []  # per-frame x within this run
        buffer_pos_y = []
        buffer_intermediate = []
        buffer_terminal = []
        buffer_control = []

        # ---------- Choose predictor ---------
        obj_predictor = self.predictor
        obj_predictor_gt=Predictors(
            chosen_predictor="eigen",
            prediction_len=self.prediction_len,
            history_len=self.history_len,
            dt=self.dt,
            dataset=self.dataset_name,
            device='cpu'
        )# linear predictor as comparison

        # ---------- CP module (update once per frame) ---------
        cp_module = self.cp_module
        cp_module_gt = cp_module

        obs, simulation_info = self.env.reset()
        truncated = False
        ego = obs['ego']  # ego : ego-vehicle(or robot)
        position_x, position_y, orientation_z = ego['position_x'], ego['position_y'], ego['orientation_z']
        cmd_linear_x, cmd_angular_z = 0.0, 0.0   # last cmd
        self._ci_series.clear()
        begin = time.time()

        self.set_buffer()       

        while not truncated:
            detect_time = time.time()
            print(frame, position_x, position_y, orientation_z, cmd_linear_x, cmd_angular_z, time.time() - detect_time)

            # record robot trajectory
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # --------- Predictor (once per frame) ---------
            pred_start = time.time()
            prediction_res = obj_predictor(obs['non-ego'])
            prediction_res_gt = obj_predictor_gt(obs['non-ego'])
            pred_time = time.time() - pred_start
            self.buffer_prediction_times.append(pred_time)

            # --------- CP update (once per frame) ---------
            confidence_intervals = cp_module.update(
                obs['non-ego'],
                prediction_res
            )
            confidence_intervals_gt=cp_module_gt.update(
                obs['non-ego'],
                prediction_res_gt
            )

            # --------- Controller (once per frame, with predictions) ---------
            velocity, controller_info, minimum, intermediate, terminal, control, minimal = self.controller(
                    **obs['ego'],
                    cmd_linear_x=cmd_linear_x,
                    cmd_angular_z=cmd_angular_z,
                    boxes=self.persistent_static_boxes,
                    predictions=prediction_res,
                    confidence_intervals=confidence_intervals,
                    goal=self.goal
            )
            # For GT(Oracle based) : no status update here, just for get controller input for GT
            # TODO: implement the lagged ACI; this is cheating
            '''
            velocity_gt, controller_info_gt, minimum_gt, intermediate_gt, terminal_gt, control_gt, minimal_gt = controller_gt(
                **scene_obs['ego'],
                boxes=persistent_static_boxes,
                predictions=valid_obs_future_true,
                confidence_intervals=np.zeros(prediction_len),
                goal=goal
            )
            '''
            
            # --------- Feasibility handling ---------
            if not controller_info['feasible']:
                cmd_linear_x, cmd_angular_z = 0., 0.
                print(frame, 'No safe paths found, stopping robot movement for this frame.',
                      position_x, position_y, cmd_linear_x, cmd_angular_z, time.time() - detect_time)
            else:
                cmd_linear_x, cmd_angular_z = velocity[0]

            # --------- Apply first control step ---------
            obs, terminated, truncated, simulation_info = self.env.step(np.array([cmd_linear_x, cmd_angular_z]))
            ego = obs['ego']
            position_x, position_y, orientation_z = ego['position_x'], ego['position_y'], ego['orientation_z']
            
            # --- Ground-truth future for energy (same frame) ---
            gt_future = self.dataset_obj.get_future(
                timestep=self.env.timestep,
                future_length=self.prediction_len,
                history_length=self.history_len,
            )

            # --- Energy via score function (Eq. (2)/(3)/(4)) ---
            #   - 'traj' : no need controller component
            #   - 'control'/'obj' : need controller interface
            if self.ci_mode == "traj":
                E_t = self.sf(x=None, y_future=gt_future, yhat_future=prediction_res)
            else:
                # 'control'/'obj' : need controller API
                # action: controller(x, predictions=...) -> action(ndarray)
                # regret: controller.solve_with_cost(x, predictions=...) -> (u, J_scalar)
                # If the current does not provide the controller API, use 'traj' first
                E_t = self.sf(x=obs['ego'], y_future=gt_future, yhat_future=prediction_res, controller=self.controller)

            eps = 1e-12  # 0-division º¸È£
            ci_scheme = "pairwise"
            if ci_scheme == "static":
                # (A) »ó¼ö ÂüÁ¶Çü: E_ref¸¦ °íÁ¤ »ó¼ö·Î
                if self.E_ref_fixed is None:
                    if self.ci_boot > 0:
                        # ¿ö¹Ö¾÷ ¼öÁý Áß
                        self._E_boot.append(E_t)
                        if len(self._E_boot) < self.ci_boot:
                            # ÀÓ½Ã Ç¥½Ã(¼±ÅÃ): r_star·Î °¡´Â ÀÓ½Ã°ª
                            E_ref_now = self.r_star
                            I_t = E_ref_now / (E_t + E_ref_now + eps)
                            self._ci_series.append(I_t)
                            fig, ax = self.env.render(c=self._ci_series + [I_t])
                            continue
                        else:
                            # ºÎÆ®½ºÆ®·¦À¸·Î '°íÁ¤' E_ref °áÁ¤ (¿©ÀüÈ÷ »ó¼ö)
                            self.E_ref_fixed = float(np.quantile(self._E_boot, self.ci_ref_q))
                            if self.E_ref_fixed <= 0:
                                self.E_ref_fixed = self.r_star  # ¾ÈÀü °¡µå
                    else:
                        # ºÎÆ®½ºÆ®·¦À» ¾²Áö ¾Ê´Â °æ¿ì: Áï½Ã »ó¼ö ¼³Á¤
                        self.E_ref_fixed = float(self.r_star)

                E_ref_now = float(self.E_ref_fixed)
                I_t = E_ref_now / (E_t + E_ref_now + eps)

            elif ci_scheme == "pairwise":
                # (B) ½Ö´ë ºñ±³Çü: E_ref = E_base_t (½Ã°£¸¶´Ù ´Þ¶óÁü)  ¡æ º¸Áõ(ÇÏÇÑ) ¾øÀ½, ÇÏÁö¸¸ Æ©´× ºÒÇÊ¿ä
                if obj_predictor_gt is None:
                    raise RuntimeError("pairwise CI needs a baseline predictor (e.g., linear)")
                baseline_pred = prediction_res_gt
                E_base_t = self.sf(
                    x=None if self.ci_mode=="traj" else obs['ego'],
                    y_future=gt_future,
                    yhat_future=baseline_pred,
                    controller=self.controller if self.ci_mode!="traj" else None
                )
                I_t = E_base_t / (E_t + E_base_t + eps)

            # --- ACI upper bound for energy, then lower CI bound (paper) ---
            U_t = self.aci_energy.update(score=E_t)
            L_t = self.r_star / (U_t + self.r_star)

            print("CI_lower(L_t): ", L_t)
            print("CI(I_t): ", I_t)
            self._ci_series.append(I_t)

            # color gradation history
            c_2 = self._ci_series.copy()
            c_2.append(I_t)

            '''
            ci = (confidence_intervals_gt[8]) / (confidence_intervals[8] + confidence_intervals_gt[8])
            print("CI: ", ci)
            self._ci_series.append(ci)
            c_2 = self._ci_series.copy()
            c_2.append(ci)
            '''

            # --------- Rendering the situation ----------
            #fig, ax = self.env.render(c=c_2)
            fig, ax = self.env.render(c=self._ci_series + [float(I_t)])

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())  # shape : (h, w, 4), RGBA
            rgb  = rgba[:, :, :3].copy()                 # drop alpha
            frames.append(rgb)
            fig.savefig(os.path.join('./viz_example', '{:03d}.png'.format(self.env.timestep)), bbox_inches='tight', pad_inches=0)

            # --------- Goal check ---------
            if terminated or self.env.timestep >= self.dataset_obj.max_timesteps - 1:
                print(frame, 'Goal reached!')
                is_success = True
                self.success_count += 1
                travel_time = time.time() - begin
                break

            # --------- Accumulate costs ---------
            minimum_cost.append(minimal)
            buffer_intermediate.append(intermediate)
            buffer_terminal.append(terminal)
            buffer_control.append(control)

            frame += 1

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

        #if ci_data:
        #    save_ci_traj_positions_csv(iter_out_dir=iter_out_dir, iteration_index=times+1, rows=ci_data)

        # ---- Iteration-level rates and summaries ----
        self.buffer_collision_rate.append(collision_count / max(1, frame))
        self.buffer_infeasible_rate.append(infeasible_count / max(1, frame))

        print("Next : #{}_scenario".format(times + 1))
        print("Collision_rate: ", collision_count / max(1, frame))
        print("Infeasible_rate: ", infeasible_count / max(1, frame))
        if self.buffer_prediction_times:
            print("Avg_prediction_time: ", np.sum(self.buffer_prediction_times) / len(self.buffer_prediction_times))
            print('Variance prediction time', np.var(self.buffer_prediction_times))
        if is_success and minimum_cost:
            print("Avg_minimal_cost: ", np.sum(minimum_cost) / len(minimum_cost))
            print("Avg_intermediate_cost: ", np.sum(buffer_intermediate) / len(buffer_intermediate) if buffer_intermediate else np.nan)
            print("Avg_terminal_cost: ", np.sum(buffer_terminal) / len(buffer_terminal) if buffer_terminal else np.nan)
            print("Avg_control_cost: ", np.sum(buffer_control) / len(buffer_control) if buffer_control else np.nan)
            print("Travel_time: ", travel_time)





