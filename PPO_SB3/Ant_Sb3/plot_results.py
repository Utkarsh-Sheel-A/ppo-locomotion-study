"""
Plots training reward, evaluation reward (with std band), and evaluation
episode length for a completed (or in-progress) Walker2d run.

Reads:
  - Monitor CSVs at logs/walker2d_ppo/monitor/  (training curve)
  - evaluations.npz at logs/walker2d_ppo/       (written automatically by EvalCallback)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

run_name = "ant_ppo_2048_lr3e-04"
log_dir = f"./logs/{run_name}/"
monitor_dir = os.path.join(log_dir, "monitor")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# --- Training reward curve (from Monitor CSVs) ---
try:
    results = load_results(monitor_dir)
    x, y = ts2xy(results, "timesteps")
    # simple moving average to smooth the very noisy per-episode signal
    window = 50
    if len(y) >= window:
        y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
        x_smooth = x[window - 1:]
    else:
        x_smooth, y_smooth = x, y
    axes[0].plot(x, y, alpha=0.25, color="tab:blue", label="raw episode reward")
    axes[0].plot(x_smooth, y_smooth, color="tab:blue", label=f"{window}-episode moving avg")
    axes[0].set_title("Training reward (rollout)")
    axes[0].legend(fontsize=8)
except Exception as e:
    axes[0].set_title("Training reward (no data yet)")
    print(f"Could not load monitor logs: {e}")

# --- Evaluation reward + episode length (from EvalCallback's saved npz) ---
eval_path = os.path.join(log_dir, "evaluations.npz")
try:
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]        # shape: (n_evals, n_eval_episodes)
    ep_lengths = data["ep_lengths"]  # shape: (n_evals, n_eval_episodes)

    mean_reward = results.mean(axis=1)
    std_reward = results.std(axis=1)
    mean_length = ep_lengths.mean(axis=1)

    axes[1].plot(timesteps, mean_reward, color="tab:green")
    axes[1].fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward,
                          alpha=0.2, color="tab:green")
    axes[1].set_title("Eval reward (mean +/- std)")

    axes[2].plot(timesteps, mean_length, color="tab:orange")
    axes[2].set_title("Eval episode length")
except FileNotFoundError:
    axes[1].set_title("Eval reward (no data yet)")
    axes[2].set_title("Eval episode length (no data yet)")
    print(f"No evaluations.npz found at {eval_path} yet - run training first.")

for ax in axes:
    ax.set_xlabel("timesteps")
    ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(log_dir, "training_curves.png")
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
plt.show()
