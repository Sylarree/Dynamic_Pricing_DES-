import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# try importing the provided codebase
try:
    import config
    from queues import InputQueue, ServiceQueue
    CODEBASE_OK = True
except Exception as e:
    CODEBASE_OK = False
    CODEBASE_ERR = str(e)

# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Fair Dynamic Pricing - Nonlinear Lens", layout="wide")

st.title("Fair Dynamic Pricing as a Nonlinear Queueing-Control Problem")
st.caption("Interactive simulation + nonlinear systems interpretation")

# =========================================================
# Default constants
# =========================================================
EPS = 1e-8

DEFAULTS = {
    "t_step": 0.1,
    "sim_time": 100.0,
    "u_min": 0.0,
    "u_max": 20.0,
    "beta_star": 0.5,
    "beta_jam": 0.1,
    "x_critic_for_beta": 0.7,
    "x_max": 1.0,
    "x_star": 0.8,
    "tol_par": 0.65,
    "lambda1": 0.7,
    "lambda2": 0.3,
    "s1": 1.0,
    "s2": 0.1,
    "z1_0": 0.2,
    "z2_0": 0.2,
    "q1_0": 0.2,
    "q2_0": 0.4,
    "state_scale": 100,
}

if CODEBASE_OK:
    DEFAULTS["t_step"] = getattr(config, "t_step", DEFAULTS["t_step"])
    DEFAULTS["sim_time"] = getattr(config, "sim_time", DEFAULTS["sim_time"])
    DEFAULTS["u_min"] = getattr(config, "u_min", DEFAULTS["u_min"])
    DEFAULTS["u_max"] = getattr(config, "u_max", DEFAULTS["u_max"])
    DEFAULTS["beta_star"] = getattr(config, "beta_star", DEFAULTS["beta_star"])
    DEFAULTS["beta_jam"] = getattr(config, "beta_jam", DEFAULTS["beta_jam"])
    DEFAULTS["x_max"] = getattr(config, "x_max", DEFAULTS["x_max"])
    DEFAULTS["x_star"] = getattr(config, "x_star", DEFAULTS["x_star"])

# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Simulation controls")

sim_mode = st.sidebar.selectbox(
    "Simulation mode",
    ["Mean-field / code-consistent", "DES-inspired stochastic"]
)

policy_mode = st.sidebar.selectbox(
    "Policy",
    ["none", "surge", "controller"]
)

compare_all = st.sidebar.checkbox("Compare all three policies", value=True)

sim_time = st.sidebar.slider("Simulation horizon", 20.0, 150.0, float(DEFAULTS["sim_time"]), 5.0)
t_step = st.sidebar.slider("Time step", 0.02, 0.5, float(DEFAULTS["t_step"]), 0.02)

tol_par = st.sidebar.slider("Tolerance parameter ρ", 0.1, 1.2, float(DEFAULTS["tol_par"]), 0.05)
x_star = st.sidebar.slider("Target service queue q*", 0.2, 1.0, float(DEFAULTS["x_star"]), 0.05)
x_critic = st.sidebar.slider("Critical congestion q_c", 0.2, 1.0, float(DEFAULTS["x_critic_for_beta"]), 0.05)
x_max = st.sidebar.slider("Queue cap x_max", 0.8, 2.0, float(DEFAULTS["x_max"]), 0.1)

beta_star = st.sidebar.slider("β*", 0.1, 1.5, float(DEFAULTS["beta_star"]), 0.05)
beta_jam = st.sidebar.slider("β_jam", 0.01, 0.5, float(DEFAULTS["beta_jam"]), 0.01)

lambda1 = st.sidebar.slider("Class 1 arrival level λ1", 0.1, 1.5, float(DEFAULTS["lambda1"]), 0.05)
lambda2 = st.sidebar.slider("Class 2 arrival level λ2", 0.1, 1.5, float(DEFAULTS["lambda2"]), 0.05)

s1 = st.sidebar.slider("Class 1 price sensitivity s1", 0.0, 3.0, float(DEFAULTS["s1"]), 0.05)
s2 = st.sidebar.slider("Class 2 price sensitivity s2", 0.0, 3.0, float(DEFAULTS["s2"]), 0.05)

z1_0 = st.sidebar.slider("Initial z1", 0.0, 2.0, float(DEFAULTS["z1_0"]), 0.05)
z2_0 = st.sidebar.slider("Initial z2", 0.0, 2.0, float(DEFAULTS["z2_0"]), 0.05)
q1_0 = st.sidebar.slider("Initial q1", 0.0, 1.5, float(DEFAULTS["q1_0"]), 0.05)
q2_0 = st.sidebar.slider("Initial q2", 0.0, 1.5, float(DEFAULTS["q2_0"]), 0.05)

u_max = st.sidebar.slider("Max price/control", 1.0, 30.0, float(DEFAULTS["u_max"]), 1.0)

demand_profile = st.sidebar.selectbox(
    "Demand scenario",
    ["constant", "piecewise peak", "sinusoidal"]
)

state_scale = st.sidebar.slider(
    "DES-inspired event scale",
    20, 500, int(DEFAULTS["state_scale"]), 10,
    help="Only used in DES-inspired stochastic mode"
)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=1)

play_slider = st.sidebar.checkbox("Show moving point on phase plane", value=True)

params = {
    "sim_time": sim_time,
    "t_step": t_step,
    "tol_par": tol_par,
    "x_star": x_star,
    "x_critic_for_beta": x_critic,
    "x_max": x_max,
    "beta_star": beta_star,
    "beta_jam": beta_jam,
    "lambda1": lambda1,
    "lambda2": lambda2,
    "s1": s1,
    "s2": s2,
    "z1_0": z1_0,
    "z2_0": z2_0,
    "q1_0": q1_0,
    "q2_0": q2_0,
    "u_min": 0.0,
    "u_max": u_max,
    "state_scale": state_scale,
    "seed": seed,
}

params["m"] = (beta_star - beta_jam) / max(x_max - x_critic, 1e-8)

# =========================================================
# Core nonlinear quantities
# =========================================================
def tolerance_ratio(q, rho):
    return max(0.0, 1.0 - rho * q)

def beta_total(q, p):
    if q <= p["x_critic_for_beta"]:
        return p["beta_star"] / max(p["x_critic_for_beta"], EPS) * q
    return max(
        p["beta_jam"],
        p["m"] * (p["x_max"] - q) + p["beta_jam"]
    )

def demand_inputs(step, p):
    if demand_profile == "constant":
        return p["lambda1"], p["lambda2"]
    if demand_profile == "piecewise peak":
        if 0.35 * p["sim_time"] <= step * p["t_step"] <= 0.55 * p["sim_time"]:
            return min(1.5, p["lambda1"] + 0.2), min(1.5, p["lambda2"] + 0.1)
        return p["lambda1"], p["lambda2"]
    if demand_profile == "sinusoidal":
        t = step * p["t_step"]
        l1 = max(0.0, p["lambda1"] + 0.15 * np.sin(t / 3))
        l2 = max(0.0, p["lambda2"] + 0.10 * np.cos(t / 4))
        return l1, l2
    return p["lambda1"], p["lambda2"]

def alpha_components(z1, z2, q, u, lam1, lam2, p):
    r = tolerance_ratio(q, p["tol_par"])
    a1 = max(0.0, r * z1 - p["s1"] * r * lam1 * u)
    a2 = max(0.0, r * z2 - p["s2"] * r * lam2 * u)
    return a1, a2

def reduced_dynamics(z, q, u, lam1, lam2, mix1, p):
    z1 = mix1 * z
    z2 = (1.0 - mix1) * z
    r = tolerance_ratio(q, p["tol_par"])
    zdot = (lam1 + lam2) - r * z
    a1, a2 = alpha_components(z1, z2, q, u, lam1, lam2, p)
    qdot = (a1 + a2) - beta_total(q, p)
    return zdot, qdot

# =========================================================
# Policy definitions
# =========================================================
def policy_none(z1, z2, q1, q2, lam1, lam2, p):
    return 0.0

def policy_surge(z1, z2, q1, q2, lam1, lam2, p):
    q = q1 + q2
    rho = min(1.0, q / max(p["x_max"], EPS))
    u = p["u_max"] * (rho ** 2)
    return float(np.clip(u, p["u_min"], p["u_max"]))

def policy_controller(z1, z2, q1, q2, lam1, lam2, p):
    # try using provided controller
    if CODEBASE_OK:
        try:
            iq1 = InputQueue(idx=0, tol_par=p["tol_par"], price_sensitivity=p["s1"], init_state=z1)
            iq2 = InputQueue(idx=1, tol_par=p["tol_par"], price_sensitivity=p["s2"], init_state=z2)
            sq = ServiceQueue(x_star=p["x_star"], init_x_list=[q1, q2])
            sq.x_critic_for_beta = p["x_critic_for_beta"]
            sq.x_max = p["x_max"]
            sq.beta_star = p["beta_star"]
            sq.beta_jam = p["beta_jam"]
            sq.input_queues = [iq1, iq2]
            r1 = iq1.get_tolerance_ratio(q1 + q2)
            r2 = iq2.get_tolerance_ratio(q1 + q2)
            u, _ = sq.cal_u_1([p["s1"], p["s2"]], [r1, r2])
            return float(np.clip(u, p["u_min"], p["u_max"]))
        except Exception:
            pass

    # fallback heuristic if Gurobi/codebase not available
    q = q1 + q2
    z = z1 + z2
    gap = q - p["x_star"]
    hetero = abs(p["s1"] - p["s2"])
    u = 3.0 * max(0.0, gap) + 0.8 * hetero + 0.5 * max(0.0, z - 1.0)
    return float(np.clip(u, p["u_min"], p["u_max"]))

POLICY_FUNCS = {
    "none": policy_none,
    "surge": policy_surge,
    "controller": policy_controller,
}

# =========================================================
# Simulation engines
# =========================================================
def simulate_mean_field(policy_name, p):
    np.random.seed(p["seed"])
    policy = POLICY_FUNCS[policy_name]

    steps = int(p["sim_time"] / p["t_step"])
    t = np.arange(steps + 1) * p["t_step"]

    z1, z2 = p["z1_0"], p["z2_0"]
    q1, q2 = p["q1_0"], p["q2_0"]

    data = []

    revenue_cum = 0.0
    unfair_integral = 0.0

    for k in range(steps + 1):
        q = q1 + q2
        z = z1 + z2
        lam1, lam2 = demand_inputs(k, p)

        u = policy(z1, z2, q1, q2, lam1, lam2, p)
        r = tolerance_ratio(q, p["tol_par"])
        a1, a2 = alpha_components(z1, z2, q, u, lam1, lam2, p)
        beta = beta_total(q, p)

        if q > EPS:
            b1 = beta * q1 / q
            b2 = beta * q2 / q
        else:
            b1, b2 = 0.0, 0.0

        # code-consistent mean-field queue updates
        z1_dot = lam1 - r * z1
        z2_dot = lam2 - r * z2
        q1_dot = a1 - b1
        q2_dot = a2 - b2

        phi1 = a1 / (z1 + EPS)
        phi2 = a2 / (z2 + EPS)
        fairness_gap = abs(phi1 - phi2)
        fairness_ratio = phi1 / (phi2 + EPS)

        unfair_integral += fairness_gap * p["t_step"]
        revenue_inst = u * (a1 + a2) * z
        revenue_cum += revenue_inst * p["t_step"]

        data.append({
            "t": k * p["t_step"],
            "z1": z1, "z2": z2, "z": z,
            "q1": q1, "q2": q2, "q": q,
            "u": u, "r": r,
            "a1": a1, "a2": a2, "alpha": a1 + a2,
            "b1": b1, "b2": b2, "beta": beta,
            "phi1": phi1, "phi2": phi2,
            "fairness_gap": fairness_gap,
            "fairness_ratio": fairness_ratio,
            "revenue_inst": revenue_inst,
            "revenue_cum": revenue_cum,
            "unfair_integral": unfair_integral,
        })

        if k < steps:
            z1 = max(0.0, z1 + p["t_step"] * z1_dot)
            z2 = max(0.0, z2 + p["t_step"] * z2_dot)
            q1 = max(0.0, q1 + p["t_step"] * q1_dot)
            q2 = max(0.0, q2 + p["t_step"] * q2_dot)

    return pd.DataFrame(data)

def simulate_des_like(policy_name, p):
    np.random.seed(p["seed"])
    policy = POLICY_FUNCS[policy_name]
    N = p["state_scale"]
    steps = int(p["sim_time"] / p["t_step"])

    # integer counts internally, normalized externally
    Z1 = int(round(p["z1_0"] * N))
    Z2 = int(round(p["z2_0"] * N))
    Q1 = int(round(p["q1_0"] * N))
    Q2 = int(round(p["q2_0"] * N))

    data = []
    revenue_cum = 0.0
    unfair_integral = 0.0

    for k in range(steps + 1):
        z1, z2 = Z1 / N, Z2 / N
        q1, q2 = Q1 / N, Q2 / N
        q = q1 + q2
        z = z1 + z2

        lam1, lam2 = demand_inputs(k, p)
        u = policy(z1, z2, q1, q2, lam1, lam2, p)
        r = tolerance_ratio(q, p["tol_par"])
        a1_rate, a2_rate = alpha_components(z1, z2, q, u, lam1, lam2, p)
        beta = beta_total(q, p)

        if q > EPS:
            b1_rate = beta * q1 / q
            b2_rate = beta * q2 / q
        else:
            b1_rate, b2_rate = 0.0, 0.0

        # eventized approximation
        arr1 = np.random.poisson(max(lam1, 0.0) * N * p["t_step"])
        arr2 = np.random.poisson(max(lam2, 0.0) * N * p["t_step"])

        adm1 = min(Z1 + arr1, np.random.poisson(max(a1_rate, 0.0) * N * p["t_step"]))
        adm2 = min(Z2 + arr2, np.random.poisson(max(a2_rate, 0.0) * N * p["t_step"]))

        dep1 = min(Q1 + adm1, np.random.poisson(max(b1_rate, 0.0) * N * p["t_step"]))
        dep2 = min(Q2 + adm2, np.random.poisson(max(b2_rate, 0.0) * N * p["t_step"]))

        Z1 = max(0, Z1 + arr1 - adm1)
        Z2 = max(0, Z2 + arr2 - adm2)
        Q1 = max(0, Q1 + adm1 - dep1)
        Q2 = max(0, Q2 + adm2 - dep2)

        phi1 = a1_rate / (z1 + EPS)
        phi2 = a2_rate / (z2 + EPS)
        fairness_gap = abs(phi1 - phi2)
        fairness_ratio = phi1 / (phi2 + EPS)

        unfair_integral += fairness_gap * p["t_step"]
        revenue_inst = u * (a1_rate + a2_rate) * z
        revenue_cum += revenue_inst * p["t_step"]

        data.append({
            "t": k * p["t_step"],
            "z1": z1, "z2": z2, "z": z,
            "q1": q1, "q2": q2, "q": q,
            "u": u, "r": r,
            "a1": a1_rate, "a2": a2_rate, "alpha": a1_rate + a2_rate,
            "b1": b1_rate, "b2": b2_rate, "beta": beta,
            "phi1": phi1, "phi2": phi2,
            "fairness_gap": fairness_gap,
            "fairness_ratio": fairness_ratio,
            "revenue_inst": revenue_inst,
            "revenue_cum": revenue_cum,
            "unfair_integral": unfair_integral,
        })

    return pd.DataFrame(data)

def run_policy(policy_name, p):
    if sim_mode == "Mean-field / code-consistent":
        return simulate_mean_field(policy_name, p)
    return simulate_des_like(policy_name, p)

# =========================================================
# Plot helpers
# =========================================================
def plot_single_run(df, title_prefix=""):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    axs[0, 0].plot(df["t"], df["q"], label="q")
    axs[0, 0].plot(df["t"], df["z"], label="z")
    axs[0, 0].axhline(params["x_critic_for_beta"], linestyle="--", color="gray", label=r"$q_c$")
    axs[0, 0].axhline(params["x_star"], linestyle=":", color="black", label=r"$q^\star$")
    axs[0, 0].set_title("Aggregate states")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(df["t"], df["u"], label="u(t)")
    axs[0, 1].plot(df["t"], df["r"], label="r(q)")
    axs[0, 1].set_title("Control and tolerance ratio")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(df["t"], df["fairness_gap"], label=r"$|\phi_1-\phi_2|$")
    axs[1, 0].plot(df["t"], df["phi1"], label=r"$\phi_1$")
    axs[1, 0].plot(df["t"], df["phi2"], label=r"$\phi_2$")
    axs[1, 0].set_title("Fairness proxy")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(df["t"], df["revenue_cum"], label="cum revenue proxy")
    axs[1, 1].plot(df["t"], df["beta"], label=r"$\beta(q)$")
    axs[1, 1].plot(df["t"], df["alpha"], label=r"$\alpha(z,q,u)$")
    axs[1, 1].set_title("Flow and revenue")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    fig.suptitle(title_prefix, fontsize=15)
    plt.tight_layout()
    return fig

def plot_classwise(df, title_prefix=""):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(df["t"], df["z1"], label="z1")
    axs[0].plot(df["t"], df["z2"], label="z2")
    axs[0].plot(df["t"], df["q1"], linestyle="--", label="q1")
    axs[0].plot(df["t"], df["q2"], linestyle="--", label="q2")
    axs[0].set_title("Class-wise states")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df["t"], df["a1"], label="a1")
    axs[1].plot(df["t"], df["a2"], label="a2")
    axs[1].plot(df["t"], df["phi1"], label="phi1 = a1/z1")
    axs[1].plot(df["t"], df["phi2"], label="phi2 = a2/z2")
    axs[1].set_title("Admissions and class-wise admission fractions")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(title_prefix, fontsize=15)
    plt.tight_layout()
    return fig

def plot_policy_comparison(results):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    for name, df in results.items():
        axs[0, 0].plot(df["t"], df["q"], label=name)
        axs[0, 1].plot(df["t"], df["u"], label=name)
        axs[1, 0].plot(df["t"], df["fairness_gap"], label=name)
        axs[1, 1].plot(df["t"], df["revenue_cum"], label=name)

    axs[0, 0].axhline(params["x_critic_for_beta"], linestyle="--", color="gray", label=r"$q_c$")
    axs[0, 0].axhline(params["x_star"], linestyle=":", color="black", label=r"$q^\star$")

    axs[0, 0].set_title("Service queue q(t)")
    axs[0, 1].set_title("Price / control u(t)")
    axs[1, 0].set_title("Fairness proxy gap")
    axs[1, 1].set_title("Cumulative revenue proxy")

    for ax in axs.ravel():
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    return fig

def vector_field_plot(policy_name, p, mix1=0.7, df=None, frame_idx=None):
    z_vals = np.linspace(0.0, 3.0, 24)
    q_vals = np.linspace(0.0, p["x_max"] * 1.2, 24)
    Z, Q = np.meshgrid(z_vals, q_vals)
    U = np.zeros_like(Z)
    V = np.zeros_like(Q)

    lam1 = p["lambda1"]
    lam2 = p["lambda2"]

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]
            q = Q[i, j]
            z1 = mix1 * z
            z2 = (1 - mix1) * z
            q1 = mix1 * q
            q2 = (1 - mix1) * q
            u = POLICY_FUNCS[policy_name](z1, z2, q1, q2, lam1, lam2, p)
            zdot, qdot = reduced_dynamics(z, q, u, lam1, lam2, mix1, p)
            U[i, j] = zdot
            V[i, j] = qdot

    speed = np.sqrt(U**2 + V**2) + 1e-8
    U_n = U / speed
    V_n = V / speed

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(Z, Q, U_n, V_n, speed, alpha=0.8)
    ax.axhline(p["x_critic_for_beta"], linestyle="--", color="gray", label=r"$q_c$")
    ax.axhline(p["x_star"], linestyle=":", color="black", label=r"$q^\star$")
    ax.set_xlabel("z = z1 + z2")
    ax.set_ylabel("q = q1 + q2")
    ax.set_title(f"Reduced phase portrait under policy = {policy_name}")

    if df is not None:
        ax.plot(df["z"], df["q"], color="red", linewidth=2, label="trajectory")
        if frame_idx is not None:
            frame_idx = int(np.clip(frame_idx, 0, len(df) - 1))
            ax.scatter(df["z"].iloc[frame_idx], df["q"].iloc[frame_idx],
                       color="black", s=80, zorder=5, label="current point")

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_nonlinear_mechanisms(p):
    q_grid = np.linspace(0, p["x_max"] * 1.3, 400)
    r_grid = np.array([tolerance_ratio(q, p["tol_par"]) for q in q_grid])
    beta_grid = np.array([beta_total(q, p) for q in q_grid])

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(q_grid, r_grid)
    axs[0].axvline(p["x_critic_for_beta"], linestyle="--", color="gray", label=r"$q_c$")
    axs[0].set_title(r"Tolerance ratio $r(q)=\max(0,1-\rho q)$")
    axs[0].set_xlabel("q")
    axs[0].set_ylabel("r(q)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(q_grid, beta_grid)
    axs[1].axvline(p["x_critic_for_beta"], linestyle="--", color="gray", label=r"$q_c$")
    axs[1].set_title(r"Service law $\beta(q)$")
    axs[1].set_xlabel("q")
    axs[1].set_ylabel(r"$\beta(q)$")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    return fig

# =========================================================
# Main app body
# =========================================================
if not CODEBASE_OK:
    st.warning("Could not fully import the provided codebase. The app will still run using the fallback controller heuristic.")
    st.code(CODEBASE_ERR)

st.subheader("Nonlinear model used in the app")
st.latex(r"""
r(q)=\max(0,1-\rho q)
""")
st.latex(r"""
\dot z = (\lambda_1+\lambda_2)-r(q)\,z
""")
st.latex(r"""
\alpha_i(z_i,q,u)=\max\!\big(0,\ r(q)z_i-s_i\,r(q)\lambda_i\,u\big)
""")
st.latex(r"""
\dot q = \alpha_1+\alpha_2-\beta(q)
""")
st.latex(r"""
\beta(q)=
\begin{cases}
\dfrac{\beta^\star}{q_c}q, & q\le q_c\\[0.15cm]
\max(\beta_{\mathrm{jam}},\,m(x_{\max}-q)+\beta_{\mathrm{jam}}), & q>q_c
\end{cases}
""")

st.markdown("""
**Why this is nonlinear**
- \(r(q)\) couples downstream congestion to upstream retention
- \(\alpha_i\) mixes state and control multiplicatively
- \(\beta(q)\) changes regime at \(q_c\)
- two classes interact through the shared bottleneck \(q=q_1+q_2\)
""")

tab1, tab2, tab3 = st.tabs(["Single policy run", "Policy comparison", "Nonlinear lens"])

with tab1:
    df = run_policy(policy_mode, params)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final q", f"{df['q'].iloc[-1]:.3f}")
    c2.metric("Final z", f"{df['z'].iloc[-1]:.3f}")
    c3.metric("Mean fairness gap", f"{df['fairness_gap'].mean():.3f}")
    c4.metric("Final revenue proxy", f"{df['revenue_cum'].iloc[-1]:.2f}")

    st.pyplot(plot_single_run(df, title_prefix=f"Policy = {policy_mode} | Mode = {sim_mode}"))
    st.pyplot(plot_classwise(df, title_prefix=f"Policy = {policy_mode}"))

with tab2:
    if compare_all:
        results = {name: run_policy(name, params) for name in ["none", "surge", "controller"]}
        st.pyplot(plot_policy_comparison(results))

        summary = []
        for name, dfi in results.items():
            summary.append({
                "policy": name,
                "final_q": dfi["q"].iloc[-1],
                "final_z": dfi["z"].iloc[-1],
                "mean_fairness_gap": dfi["fairness_gap"].mean(),
                "final_revenue_proxy": dfi["revenue_cum"].iloc[-1],
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
    else:
        st.info("Enable 'Compare all three policies' in the sidebar.")

with tab3:
    st.markdown("### Nonlinear mechanisms")
    st.pyplot(plot_nonlinear_mechanisms(params))

    st.markdown("### Reduced phase portrait in the $(z,q)$ plane")
    mix1 = st.slider("Assumed class-1 share in reduced phase portrait", 0.1, 0.9, 0.7, 0.05)

    df_phase = run_policy(policy_mode, params)
    frame_idx = 0
    if play_slider:
        frame_idx = st.slider("Time index for moving point", 0, len(df_phase) - 1, min(20, len(df_phase) - 1), 1)

    st.pyplot(vector_field_plot(policy_mode, params, mix1=mix1, df=df_phase, frame_idx=frame_idx))

    st.markdown("### Suggested interpretation")
    st.markdown(r"""
- If the trajectory is pushed toward low \(q\), the policy is strongly congestion-suppressing.
- If the trajectory stays near \(q^\star\), the policy behaves like queue regulation.
- If \(z\) grows while \(q\) stays bounded, the controller is regulating the service bottleneck while allowing upstream accumulation.
- If the fairness gap stays high, one class is persistently admitted at a different rate than the other.
""")