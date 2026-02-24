#!/usr/bin/env python3

"""
Bayesian Optimization for LQR Q and R tuning
Cart-Pole under earthquake disturbance
"""

import os
import time
import signal
import subprocess
import numpy as np
from dataclasses import dataclass, asdict
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


# ==============================
# CONFIG
# ==============================

LAUNCH_CMD = (
    "source ~/ros2_ws/install/setup.bash && "
    "ros2 launch cart_pole_optimal_control cart_pole_rviz.launch.py "
    "controller_q_diag:='{qdiag}' controller_r:={r}"
)

EVAL_TIME = 35.0
TIMEOUT = 70.0

CART_LIMIT = 2.5
POLE_LIMIT_DEG = 45.0

N_CALLS = 25
N_INITIAL = 7

HISTORY_FILE = os.path.expanduser("~/cartpole_bayes_history.json")


# ==============================
# Metrics container
# ==============================

@dataclass
class Metrics:
    duration: float
    max_cart: float
    max_pole: float
    rms_cart: float
    rms_pole: float
    rms_u: float
    violated: bool


# ==============================
# ROS metric collector
# ==============================

class Collector(Node):
    def __init__(self):
        super().__init__("optimizer_metrics")

        self.x = []
        self.theta = []
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

        x = msg.position[cart_idx]
        theta_deg = np.degrees(msg.position[pole_idx])

        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9

        self.x.append(x)
        self.theta.append(theta_deg)
        self.u.append(self.current_u)

        if abs(x) > CART_LIMIT or abs(theta_deg) > POLE_LIMIT_DEG:
            self.violation = True

    def compute_metrics(self):
        if len(self.x) == 0:
            return Metrics(0, 999, 999, 999, 999, 999, True)

        xs = np.array(self.x)
        th = np.array(self.theta)
        u = np.array(self.u)

        duration = (
            self.get_clock().now().nanoseconds / 1e9
            - self.start_time
            if self.start_time else 0
        )

        return Metrics(
            duration,
            np.max(np.abs(xs)),
            np.max(np.abs(th)),
            np.sqrt(np.mean(xs**2)),
            np.sqrt(np.mean(th**2)),
            np.sqrt(np.mean(u**2)),
            self.violation
        )


# ==============================
# Cost function
# ==============================

def compute_cost(m: Metrics):
    cost = (
        2.5 * (m.rms_pole / 10)
        + 1.5 * (m.rms_cart)
        + 0.05 * m.rms_u
        - 0.1 * m.duration
    )

    if m.max_cart > CART_LIMIT:
        cost += 300

    if m.max_pole > POLE_LIMIT_DEG:
        cost += 300

    if m.violated:
        cost += 500

    return max(cost, 0)


# ==============================
# Run single trial
# ==============================

def run_trial(q, r):
    q_str = f"[{q[0]},{q[1]},{q[2]},{q[3]}]"
    cmd = LAUNCH_CMD.format(qdiag=q_str, r=r)

    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    rclpy.init()
    node = Collector()

    start = time.time()

    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)

            elapsed = time.time() - start
            if elapsed > EVAL_TIME:
                break
            if elapsed > TIMEOUT:
                break
            if node.violation and elapsed > 1:
                break

        metrics = node.compute_metrics()
        cost = compute_cost(metrics)
        return cost, metrics

    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass

        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            time.sleep(1)
        if proc.poll() is None:
            proc.kill()


# ==============================
# Bayesian optimizer
# ==============================

def main():

    space = [
        Real(0.1, 30.0, name="qx"),
        Real(0.1, 30.0, name="qxd"),
        Real(1.0, 250.0, name="qth"),
        Real(1.0, 250.0, name="qthd"),
        Real(0.01, 5.0, name="r"),
    ]

    history = []

    @use_named_args(space)
    def objective(qx, qxd, qth, qthd, r):
        q = [qx, qxd, qth, qthd]
        c, m = run_trial(q, r)

        print(f"\nTrial Q={q}, R={r:.3f} => cost={c:.3f}")

        history.append({
            "Q": q,
            "R": r,
            "cost": c,
            "metrics": asdict(m)
        })

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        return c

    result = gp_minimize(
        objective,
        space,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        random_state=42
    )

    print("\n===== BEST RESULT =====")
    print("Q =", result.x[:4])
    print("R =", result.x[4])
    print("Cost =", result.fun)


if __name__ == "__main__":
    main()