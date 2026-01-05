import numpy as np
import casadi as ca

from canvas.envs.env_utils import Geometry



def solve(ts, dt, ulb, uub,
          geometry: Geometry, predictions, r_robot, r_agent,
          initial_pose, goal,
          X, U
          ):
    """
    ts: timesteps; [0, ..., ts] denotes the planning horizon
    dt: sampling time
    xlb, xub, ulb, uub: 1-dim. numpy arrays representing the state & ctrl bounds of the ego-robot
    goal: numpy array of shape (2,) representing the goal position
    """

    # state & ctrl dim.
    dx = 3  # (x, y, theta)
    du = 2  # (v, w)

    # optimization variables
    # the following indexing used here:
    # (i, t) -> i + d_x * t
    # where i: state idx / t: time
    x = ca.MX.sym('x', dx * ts)  # states
    u = ca.MX.sym('u', du * (ts - 1))  # ctrls

    def unicycle_dynamics(x1, u1):
        # dynamics function for the unicycle model
        v, w = u1[0], u1[1]
        th = x1[2]
        dx1 = ca.vertcat(v * ca.cos(th), v * ca.sin(th), w)
        return x1 + dx1 * dt

    # constraints & obj.
    g = []  # list of constraints
    lbg = []  # lower bounds
    ubg = []  # upper bounds
    objective = 0.

    eq_tol = 1e-6   # tolerance for equality constraints

    # init state constraints
    x0 = x[:dx]
    g.append(x0 - initial_pose)
    lbg += [-eq_tol for _ in range(dx)]
    ubg += [eq_tol for _ in range(dx)]


    for t in range(ts - 1):
        xt_begin = dx * t
        ut_begin = du * t
        xt_next_begin = xt_begin + dx
        xt = x[xt_begin: xt_begin+dx]
        ut = u[ut_begin: ut_begin+du]

        # dynamics constraints (eq.)
        xt_next = x[xt_next_begin: xt_next_begin+dx]
        g.append(xt_next - unicycle_dynamics(xt, ut))
        lbg += [-eq_tol for _ in range(dx)]
        ubg += [eq_tol for _ in range(dx)]

        pos_t_next = x[xt_next_begin: xt_next_begin+2]

        # static obstacles
        # min_i x_i -> logsumexp x
        # inverse temperature param.
        tau = 1e-1      # (tau -> 0) <=> (softmax -> max)

        for o in geometry:
            # TODO: general convex sets beyond polyhedra
            # TODO: penalty instead of constraints
            A, b = o.to_halfspaces()
            # Violation of one region: max(0, b - Ax) (element-wise)

            m = A.shape[0]  # number of constraints
            b = np.squeeze(b)
            # note that the following underapproximates the max_i {a_i x - b_i - r}
            smoothed_signed_dist = tau * (ca.logsumexp((A @ pos_t_next - b) / tau)) - tau * np.log(m)
            g.append(smoothed_signed_dist)
            lbg.append(.2)
            ubg.append(ca.inf)

        # dynamic obstacles
        for agent_id, p in predictions.items():
            t_max = p.shape[0]
            # p: numpy array of shape (ts - 1, 2)
            if t < t_max:
                signed_dist = ca.norm_2(pos_t_next - p[t]) - r_agent
                g.append(signed_dist)
                lbg.append(r_robot)
                ubg.append(ca.inf)

        # objective
        cost_weight = 10. if t == ts - 2 else 1.
        objective += cost_weight * ca.norm_2(pos_t_next - goal) ** 2 + 1e-5 * ca.norm_2(u) ** 2

    # Flatten constraints
    g = ca.vertcat(*g)

    # define the optimization problem
    opts = {'snopt.print_level': 0, 'print_time': 0, 'verbose': True}
    nlp = {'x': ca.vertcat(x, u), 'f': objective, 'g': g}
    solver = ca.nlpsol('solver', 'snopt', nlp, opts)

    # provide the initial guesses
    u_init = np.reshape(U, newshape=(-1,))
    x_init = np.reshape(X, newshape=(-1,))

    x0 = np.concatenate([x_init, u_init])
    # bounds of x_t & u_t
    xlb = np.append(geometry.lower_bound, -4. * np.pi)
    xub = np.append(geometry.upper_bound, 4. * np.pi)

    xlb_rep = np.tile(xlb, reps=ts)
    xub_rep = np.tile(xub, reps=ts)
    ulb_rep = np.tile(ulb, reps=ts-1)
    uub_rep = np.tile(uub, reps=ts-1)

    # lower bounds
    lbx = -ca.inf * ca.DM.ones(x0.shape[0])
    lbx[: dx*ts] = xlb_rep
    lbx[dx*ts:] = ulb_rep
    # upper bounds
    ubx = ca.inf * ca.DM.ones(x0.shape[0])
    ubx[: dx*ts] = xub_rep
    ubx[dx*ts:] = uub_rep


    # Solve the optimization
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    x_sol = np.reshape(sol['x'][:dx*ts], (-1, dx))
    u_sol = np.reshape(sol['x'][dx*ts:], (-1, du))

    # print("Solution:", x_sol)
    return x_sol, u_sol