import os
import numpy as np
import matplotlib.pyplot as plt

import config
from queues import InputQueue, ServiceQueue

# =========================
# Helpers
# =========================

EPS = 1e-8

def ensure_plot_dir():
    os.makedirs("./plots", exist_ok=True)

def make_system(tol_par=0.65, x_star=0.8,
                init_input=(0.2, 0.2), init_service=(0.2, 0.4),
                sensitivities=(1.0, 0.1)):
    q1 = InputQueue(idx=0, tol_par=tol_par,
                    price_sensitivity=sensitivities[0],
                    init_state=init_input[0])
    q2 = InputQueue(idx=1, tol_par=tol_par,
                    price_sensitivity=sensitivities[1],
                    init_state=init_input[1])
    sq = ServiceQueue(x_star=x_star, init_x_list=list(init_service))
    sq.input_queues = [q1, q2]
    return sq, q1, q2

def get_metrics(service_q, input_qs, u):
    z1 = input_qs[0].state
    z2 = input_qs[1].state
    z = z1 + z2

    q1, q2 = service_q.x_list
    q = q1 + q2

    a1 = input_qs[0].alpha if input_qs[0].alpha is not None else 0.0
    a2 = input_qs[1].alpha if input_qs[1].alpha is not None else 0.0
    alpha_tot = a1 + a2

    # simple revenue proxy
    revenue_inst = u * alpha_tot * z

    # fairness proxy: compare admission fractions
    phi1 = a1 / (z1 + EPS)
    phi2 = a2 / (z2 + EPS)
    fairness_gap = abs(phi1 - phi2)
    fairness_ratio = phi1 / (phi2 + EPS)

    return {
        "z1": z1, "z2": z2, "z": z,
        "q1": q1, "q2": q2, "q": q,
        "a1": a1, "a2": a2, "alpha_tot": alpha_tot,
        "u": u,
        "revenue_inst": revenue_inst,
        "fairness_gap": fairness_gap,
        "fairness_ratio": fairness_ratio,
        "phi1": phi1, "phi2": phi2,
    }

# =========================
# Policy definitions
# =========================

def policy_none(service_q, input_qs, step, params):
    return 0.0

def policy_surge(service_q, input_qs, step, params):
    q = sum(service_q.x_list)
    qmax = params.get("qmax", service_q.x_max)
    umax = params.get("umax", config.u_max)

    rho = min(1.0, q / max(qmax, 1e-8))
    # very simple monotone surge rule
    u = umax * (rho ** 2)
    return float(np.clip(u, config.u_min, config.u_max))

def policy_controller(service_q, input_qs, step, params):
    tol_ratio = [input_qs[0].get_tolerance_ratio(sum(service_q.x_list)),
                 input_qs[1].get_tolerance_ratio(sum(service_q.x_list))]
    sensitivities = [input_qs[0].price_sensitivity, input_qs[1].price_sensitivity]
    u, _ = service_q.cal_u_1(sensitivities, tol_ratio)
    return float(np.clip(u, config.u_min, config.u_max))

POLICIES = {
    "none": policy_none,
    "surge": policy_surge,
    "controller": policy_controller,
}

# =========================
# Demand profile
# =========================

def demand_profile(step, base=(0.7, 0.3), scenario="piecewise"):
    if scenario == "constant":
        return np.array(base)

    if scenario == "piecewise":
        if 80 <= step < 140:
            return np.array([0.9, 0.4])
        return np.array(base)

    if scenario == "sinusoidal":
        b1, b2 = base
        return np.array([
            np.clip(b1 + 0.15*np.sin(step/20), 0.0, 1.0),
            np.clip(b2 + 0.10*np.cos(step/25), 0.0, 1.0),
        ])

    return np.array(base)

# =========================
# Simulation runner
# =========================

def run_sim(policy_name,
            steps=None,
            tol_par=0.65,
            x_star=0.8,
            init_input=(0.2, 0.2),
            init_service=(0.2, 0.4),
            sensitivities=(1.0, 0.1),
            demand_base=(0.7, 0.3),
            demand_scenario="piecewise",
            seed=1):

    np.random.seed(seed)

    if steps is None:
        steps = int(config.sim_time / config.t_step)

    service_q, q1, q2 = make_system(
        tol_par=tol_par,
        x_star=x_star,
        init_input=init_input,
        init_service=init_service,
        sensitivities=sensitivities
    )

    log = {
        "t": [],
        "z": [], "z1": [], "z2": [],
        "q": [], "q1": [], "q2": [],
        "a1": [], "a2": [], "alpha_tot": [],
        "u": [],
        "fairness_gap": [], "fairness_ratio": [],
        "phi1": [], "phi2": [],
        "revenue_inst": [], "revenue_cum": [],
    }

    rev_cum = 0.0
    policy = POLICIES[policy_name]

    for step in range(steps):
        max_alpha = demand_profile(step, base=demand_base, scenario=demand_scenario)

        # choose control
        u = policy(service_q, [q1, q2], step, {"qmax": service_q.x_max, "umax": config.u_max})

        service_q.u_traj.append(u)

        # update input queues
        q1.update(max_alpha=max_alpha[0], x_list=service_q.x_list, u=u)
        q2.update(max_alpha=max_alpha[1], x_list=service_q.x_list, u=u)

        # update service queue
        service_q.update()

        # log
        m = get_metrics(service_q, [q1, q2], u)
        rev_cum += m["revenue_inst"] * config.t_step

        log["t"].append(step * config.t_step)
        for k in ["z", "z1", "z2", "q", "q1", "q2",
                  "a1", "a2", "alpha_tot", "u",
                  "fairness_gap", "fairness_ratio",
                  "phi1", "phi2", "revenue_inst"]:
            log[k].append(m[k])
        log["revenue_cum"].append(rev_cum)

    return log

# =========================
# Plotting
# =========================

def compare_plots(results):
    ensure_plot_dir()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for name, res in results.items():
        axs[0, 0].plot(res["t"], res["q"], label=name)
        axs[0, 1].plot(res["t"], res["u"], label=name)
        axs[1, 0].plot(res["t"], res["fairness_gap"], label=name)
        axs[1, 1].plot(res["t"], res["revenue_cum"], label=name)

    axs[0, 0].set_title("Service Queue q(t)")
    axs[0, 1].set_title("Price / Control u(t)")
    axs[1, 0].set_title("Fairness Proxy Gap |phi1-phi2|")
    axs[1, 1].set_title("Cumulative Revenue Proxy")

    axs[0, 0].set_ylabel("q")
    axs[0, 1].set_ylabel("u")
    axs[1, 0].set_ylabel("gap")
    axs[1, 1].set_ylabel("revenue")
    axs[1, 0].set_xlabel("time")
    axs[1, 1].set_xlabel("time")

    for ax in axs.ravel():
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("./plots/policy_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

def classwise_plot(results, policy_name="controller"):
    ensure_plot_dir()
    res = results[policy_name]

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs[0].plot(res["t"], res["z1"], label="z1")
    axs[0].plot(res["t"], res["z2"], label="z2")
    axs[0].plot(res["t"], res["q1"], label="q1", linestyle="--")
    axs[0].plot(res["t"], res["q2"], label="q2", linestyle="--")
    axs[0].set_title(f"Class-wise States ({policy_name})")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(res["t"], res["phi1"], label="phi1 = a1/z1")
    axs[1].plot(res["t"], res["phi2"], label="phi2 = a2/z2")
    axs[1].set_title(f"Admission Fractions ({policy_name})")
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_xlabel("time")

    plt.tight_layout()
    plt.savefig(f"./plots/{policy_name}_classwise.png", dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# Main
# =========================

if __name__ == "__main__":
    results = {}
    for name in ["none", "surge", "controller"]:
        results[name] = run_sim(
            policy_name=name,
            tol_par=0.65,
            x_star=0.8,
            init_input=(0.2, 0.2),
            init_service=(0.2, 0.4),
            sensitivities=(1.0, 0.1),
            demand_base=(0.7, 0.3),
            demand_scenario="piecewise",
            seed=1
        )

    compare_plots(results)
    classwise_plot(results, policy_name="controller")