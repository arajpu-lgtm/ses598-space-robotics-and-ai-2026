#!/usr/bin/env python3

import os
import time
import signal
import subprocess
import numpy as np
from dataclasses import dataclass, asdict
import json
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


LAUNCH_CMD = (
    "source ~/ros2_ws/install/setup.bash && "
    "ros2 launch cart_pole_optimal_control cart_pole_rviz.launch.py "
    "use_rviz:=false "
    "controller_q_diag:='{qdiag}' controller_r:={r}"
)

EVAL_TIME = 35.0
TIMEOUT = 90.0
WAIT_FOR_STATE = 15.0

CART_LIMIT = 2.5

# Quality targets (radians)
THETA_SOFT_RAD = np.deg2rad(8.0)
THETA_HARD_RAD = np.deg2rad(15.0)
EARLY_STOP_THETA_RAD = np.deg2rad(35.0)

# Bayesian Optimization budget
N_CALLS = 35
N_INITIAL = 10

# Repeat each candidate to reduce randomness
REPEATS_PER_TRIAL = 2

HISTORY_FILE = os.path.expanduser("~/cartpole_bayes_history.json")
BEST_FILE = os.path.expanduser("~/cartpole_best_qr.txt")

BASE_Q = np.array([7.0, 7.0, 24.0, 24.0], dtype=float)  
BASE_R = 0.06                                           



@dataclass
class Metrics:
    duration: float
    max_cart: float
    max_pole: float      # radians
    rms_cart: float
    rms_pole: float      # radians
    rms_u: float
    violated: bool


class Collector(Node):
    def __init__(self):
        super().__init__("optimizer_metrics")

        self.x = []
        self.theta = []   # radians
        self.u = []

        self.current_u = 0.0
        self.start_time = None
        self.violation = False

        self.create_subscription(
            JointState,
            "/world/empty/model/cart_pole/joint_state",
            self.state_callback,
            50
        )

        self.create_subscription(
            Float64,
            "/model/cart_pole/joint/cart_to_base/cmd_force",
            self.control_callback,
            50
        )

    def control_callback(self, msg):
        self.current_u = float(msg.data)

    def state_callback(self, msg):
        try:
            cart_idx = msg.name.index("cart_to_base")
            pole_idx = msg.name.index("pole_joint")
        except ValueError:
            return

        x = float(msg.position[cart_idx])
        theta = float(msg.position[pole_idx])  # radians

        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9

        self.x.append(x)
        self.theta.append(theta)
        self.u.append(self.current_u)

        if abs(x) > CART_LIMIT or abs(theta) > EARLY_STOP_THETA_RAD:
            self.violation = True

    def compute_metrics(self) -> Metrics:
        if len(self.x) == 0:
            return Metrics(0, 999, 999, 999, 999, 999, True)

        xs = np.array(self.x, dtype=float)
        th = np.array(self.theta, dtype=float)
        u = np.array(self.u, dtype=float)

        now = self.get_clock().now().nanoseconds / 1e9
        duration = (now - self.start_time) if self.start_time else 0.0

        return Metrics(
            float(duration),
            float(np.max(np.abs(xs))),
            float(np.max(np.abs(th))),
            float(np.sqrt(np.mean(xs**2))),
            float(np.sqrt(np.mean(th**2))),
            float(np.sqrt(np.mean(u**2))),
            bool(self.violation)
        )



# Cost (lower is better)

def compute_cost(m: Metrics) -> float:
    if m.duration <= 0.0 or m.max_cart > 100 or m.max_pole > 100:
        return 1e6

    vals = [m.max_cart, m.max_pole, m.rms_cart, m.rms_pole, m.rms_u]
    if any((math.isnan(v) or math.isinf(v)) for v in vals):
        return 1e6

    # theta dominates
    w_rms_theta = 1500.0
    w_max_theta = 600.0
    w_cart = 6.0
    w_u = 0.001
    w_time = -4.0

    cost = (
        w_rms_theta * (m.rms_pole ** 2) +
        w_max_theta * (m.max_pole ** 2) +
        w_cart * (m.rms_cart ** 2) +
        w_u * (m.rms_u ** 2) +
        w_time * m.duration
    )

    # capped penalties (avoid saturation)
    over_soft = max(0.0, m.max_pole - THETA_SOFT_RAD)
    cost += min(60000.0 * (over_soft ** 2), 5000.0)

    over_hard = max(0.0, m.max_pole - THETA_HARD_RAD)
    cost += min(120000.0 * (over_hard ** 2), 15000.0)

    cart_over = max(0.0, m.max_cart - 2.3)
    cost += min(25000.0 * (cart_over ** 2), 12000.0)

    if m.violated:
        cost += 500.0

    return float(max(cost, 0.0))

# Build Q/R from scales

def build_qr(scale_x: float, scale_theta: float, scale_r: float):
    Q = BASE_Q.copy()
    Q[0] *= scale_x
    Q[1] *= scale_x
    Q[2] *= scale_theta
    Q[3] *= scale_theta
    R = float(BASE_R * scale_r)
    return Q.tolist(), R

# Run simulation

def run_once(Q, R):
    q_str = f"[{Q[0]},{Q[1]},{Q[2]},{Q[3]}]"
    cmd = LAUNCH_CMD.format(qdiag=q_str, r=R)

    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True
    )

    rclpy.init()
    node = Collector()
    start = time.time()

    try:
        # Wait for first state
        t0 = time.time()
        while len(node.x) == 0 and (time.time() - t0) < WAIT_FOR_STATE:
            rclpy.spin_once(node, timeout_sec=0.1)

        if len(node.x) == 0:
            m = node.compute_metrics()
            return 1e6, m

        # Evaluate window
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            elapsed = time.time() - start

            if elapsed > EVAL_TIME:
                break
            if elapsed > TIMEOUT:
                break
            if node.violation and elapsed > 1.0:
                break

        m = node.compute_metrics()
        c = compute_cost(m)
        return c, m

    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                time.sleep(1.0)
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def run_trial(Q, R):
    costs = []
    metrics_list = []
    for _ in range(REPEATS_PER_TRIAL):
        c, m = run_once(Q, R)
        costs.append(c)
        metrics_list.append(m)

    avg_cost = float(np.mean(costs))
    best_idx = int(np.argmin(costs))
    return avg_cost, metrics_list[best_idx]



def main():
    
    space = [
        Real(0.6, 1.6, name="scale_x"),
        Real(0.6, 1.6, name="scale_theta"),
        Real(0.6, 1.6, name="scale_r"),
    ]

    history = []
    best_cost = float("inf")

    @use_named_args(space)
    def objective(scale_x, scale_theta, scale_r):
        nonlocal best_cost

        Q, R = build_qr(scale_x, scale_theta, scale_r)
        c, m = run_trial(Q, R)

        print(
            f"\nTrial scales: sx={scale_x:.3f}, sth={scale_theta:.3f}, sr={scale_r:.3f} "
            f"=> Q={Q}, R={R:.4f} | cost={c:.3f}\n"
            f"  dur={m.duration:.1f}s cart_max={m.max_cart:.2f}m "
            f"theta_max={np.degrees(m.max_pole):.2f}deg theta_rms={np.degrees(m.rms_pole):.2f}deg "
            f"u_rms={m.rms_u:.2f} violated={m.violated}",
            flush=True
        )

        rec = {
            "scale_x": float(scale_x),
            "scale_theta": float(scale_theta),
            "scale_r": float(scale_r),
            "Q": Q,
            "R": R,
            "cost": float(c),
            "metrics": asdict(m)
        }
        history.append(rec)

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        if c < best_cost:
            best_cost = c
            with open(BEST_FILE, "w") as f:
                f.write(f"Best cost: {best_cost}\n")
                f.write(f"Best scales: sx={scale_x}, sth={scale_theta}, sr={scale_r}\n")
                f.write(f"Best Q diag: {Q}\n")
                f.write(f"Best R: {R}\n")
            print(f"? NEW BEST: cost={best_cost:.3f}  Q={Q}  R={R:.4f}", flush=True)

        return float(c)

    result = gp_minimize(
        objective,
        space,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        acq_func="EI",
        random_state=42
    )

    sx, sth, sr = result.x
    Q_best, R_best = build_qr(sx, sth, sr)

    print("\n===== BEST RESULT =====")
    print("Best scales:", result.x)
    print("Q =", Q_best)
    print("R =", R_best)
    print("Cost =", result.fun)
    print("History saved to:", HISTORY_FILE)
    print("Best saved to:", BEST_FILE)


if __name__ == "__main__":
    main()