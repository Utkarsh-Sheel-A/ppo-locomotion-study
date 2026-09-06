"""
Systematic cross-run analysis for the PPO locomotion portfolio.

Ingests every run under PPO_Project/*/logs (EvalCallback evaluations.npz +
Monitor CSVs) and the TensorBoard event files, then produces tidy result
tables and all report figures:

  results/eval_curves.csv    evaluation reward curves (mean/std per point)
  results/train_curves.csv   training episode reward/length curves
  results/tb_scalars.csv     PPO internal metrics from TensorBoard
  results/runs_summary.csv   one row per run with headline statistics
  results/final_evals.csv    20-episode deterministic final evaluations
  results/summary.md         human-readable summary tables (used by REPORT.md)
  figures/*.png              all report figures

Usage:
    conda activate rl
    python analysis/analyze_ppo.py
"""
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../PPO_Project
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

ENV_OF_PROJECT = {
    "Hopper_SB3": "Hopper-v5",
    "Walker_2d_Sb3": "Walker2d-v5",
    "Ant_Sb3": "Ant-v5",
    "Humanoid_sb3": "Humanoid-v5",
}
ENV_LABEL = {"Hopper-v5": "Hopper", "Walker2d-v5": "Walker2d",
             "Ant-v5": "Ant", "Humanoid-v5": "Humanoid"}
ENV_ORDER = ["Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5"]
ENV_COLOR = {"Hopper-v5": "#1f77b4", "Walker2d-v5": "#2ca02c",
             "Ant-v5": "#ff7f0e", "Humanoid-v5": "#d62728"}
LR_COLOR = {"1e-4": "#1f77b4", "3e-4": "#d62728"}

# Per-run palette (Okabe-Ito, colorblind-safe). Within any figure, curves of the
# same learning rate stay in the same hue family (blue = 1e-4, vermillion = 3e-4)
# while every run gets a clearly distinct shade.
RUN_COLORS = {
    "hopper_ppo":                    "#0072B2",   # blue
    "walker2d_ppo":                  "#009E73",   # bluish green
    "ant_ppo_lr1e-04":               "#0072B2",   # blue
    "ant_ppo_lr3e-04":               "#D55E00",   # vermillion
    "ant_ppo_2048_lr3e-04":          "#E69F00",   # orange
    "humanoid_ppo_lr1e-04_seed42":   "#0072B2",   # blue
    "humanoid_ppo_lr1e-04_seed7":    "#56B4E9",   # sky blue
    "humanoid_ppo_lr3e-04_seed42":   "#D55E00",   # vermillion
}
DEFAULT_LINE_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def run_color(run, index=0):
    """Distinct, high-contrast color for a run's curve."""
    return RUN_COLORS.get(run["run"], DEFAULT_LINE_CYCLE[index % len(DEFAULT_LINE_CYCLE)])

# TensorBoard tags that describe PPO's training dynamics (subset kept so the
# CSV stays small; the full set is still in the event files).
TB_TAGS = [
    "rollout/ep_rew_mean", "rollout/ep_len_mean", "time/fps",
    "train/explained_variance", "train/entropy_loss", "train/clip_fraction",
    "train/approx_kl", "train/value_loss", "train/policy_gradient_loss",
    "train/loss", "train/std",
]
MILESTONES = [0.5e6, 1e6, 2e6, 3e6, 5e6]

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    # seaborn-whitegrid sets legend.frameon=False - and since style.use() applies
    # AFTER any rcParams set before it, this block must come after style.use(),
    # not before, or these overrides get silently clobbered back to the style's
    # defaults. Cheap insurance regardless of the color palette in use - force an
    # opaque legend box everywhere.
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.facecolor": "white", "legend.edgecolor": "0.8",
})


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def fmt_lr(lr: str) -> str:
    """'1e-04' -> '1e-4' for display."""
    return f"{float(lr):.0e}".replace("e-0", "e-")


def parse_run_meta(run_name: str):
    """Extract lr / seed / variant from a run folder name."""
    m = re.search(r"lr(\d(?:\.\d+)?e-?\d+)", run_name)
    lr = m.group(1) if m else "3e-4"          # Hopper/Walker hardcode 3e-4
    s = re.search(r"seed(\d+)", run_name)
    seed = int(s.group(1)) if s else 42
    variant = "n_steps=2048" if "2048" in run_name else "n_steps=1024"
    return lr, seed, variant


def load_eval_curve(log_dir):
    """-> (timesteps, eval_mean, eval_std, ep_len_mean) from evaluations.npz."""
    d = np.load(os.path.join(log_dir, "evaluations.npz"))
    res = d["results"]
    return d["timesteps"], res.mean(axis=1), res.std(axis=1), d["ep_lengths"].mean(axis=1)


def load_train_curve(log_dir):
    """-> (env_steps, ep_reward, ep_length) from all Monitor CSVs of a run."""
    rows = []
    for f in sorted(glob.glob(os.path.join(log_dir, "monitor", "*.monitor.csv"))):
        with open(f) as fh:
            next(fh)                            # JSON header line
            for line in csv.reader(fh):
                if len(line) < 3:
                    continue
                try:
                    rows.append((float(line[2]), float(line[0]), float(line[1])))
                except ValueError:
                    continue            # literal 'r,l,t' header row in some files
    rows.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in rows])
    rew = np.array([r[1] for r in rows])
    lens = np.array([r[2] for r in rows])
    return np.cumsum(lens), rew, lens           # x-axis: cumulative env steps


def monitor_t_start(log_dir):
    """Training wall-clock start from the Monitor JSON header (run fingerprint)."""
    files = sorted(glob.glob(os.path.join(log_dir, "monitor", "*.monitor.csv")))
    if not files:
        return None
    with open(files[0]) as fh:
        return json.loads(fh.readline().lstrip("#").strip()).get("t_start")


def find_event_dirs():
    """All directories that directly contain a TensorBoard events file."""
    out = []
    for pat in [os.path.join(ROOT, "*", "tb_logs", "*", "*"),
                os.path.join(ROOT, "*", "logs", "*", "PPO_*")]:
        for d in sorted(glob.glob(pat)):
            if glob.glob(os.path.join(d, "events.out.tfevents.*")):
                out.append(d)
    return out


def event_file_ts(event_dir):
    """Unix timestamp embedded in the events filename (run fingerprint)."""
    name = os.path.basename(glob.glob(os.path.join(event_dir, "events.out.tfevents.*"))[0])
    return float(name.split(".")[3])


def load_tb_scalars(event_dir):
    """-> {tag: (steps, values)} for the tags we care about."""
    acc = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        if tag in TB_TAGS:
            ev = acc.Scalars(tag)
            out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def load_final_eval(model_dir):
    path = os.path.join(model_dir, "final_eval_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def rolling_mean(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# --------------------------------------------------------------------------- #
# Run collection
# --------------------------------------------------------------------------- #
def collect_runs():
    """Discover every run (logs/<name> containing evaluations.npz)."""
    runs = []
    for project in sorted(glob.glob(os.path.join(ROOT, "*"))):
        project_name = os.path.basename(project)
        if project_name not in ENV_OF_PROJECT:
            continue
        env_id = ENV_OF_PROJECT[project_name]
        for log_dir in sorted(glob.glob(os.path.join(project, "logs", "*"))):
            if not os.path.exists(os.path.join(log_dir, "evaluations.npz")):
                continue
            run_name = os.path.basename(log_dir.rstrip("/"))
            lr, seed, variant = parse_run_meta(run_name)
            runs.append({
                "project": project_name, "env": env_id, "run": run_name,
                "lr": lr, "seed": seed, "variant": variant,
                "log_dir": log_dir,
                "model_dir": os.path.join(project, "models", run_name),
                "t_start": monitor_t_start(log_dir),
            })
    # attach the TensorBoard event dir closest in start time (same project);
    # needed because two Ant runs share the tb_log_name 'lr3e-04'.
    event_dirs = find_event_dirs()
    for run in runs:
        candidates = [d for d in event_dirs
                      if d.startswith(os.path.join(ROOT, run["project"]) + os.sep)]
        if run["t_start"] is not None and candidates:
            run["event_dir"] = min(candidates, key=lambda d: abs(event_file_ts(d) - run["t_start"]))
        elif candidates:
            run["event_dir"] = candidates[0]
        else:
            run["event_dir"] = None
    return runs


def label(run, with_seed=True):
    lab = f"lr {fmt_lr(run['lr'])}"
    if run["variant"] != "n_steps=1024":
        lab += f" ({run['variant']})"
    if with_seed:
        lab += f", seed {run['seed']}"
    return lab


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def eval_at(ts, mean, t):
    """Evaluation reward at env-step t (linear interpolation, clamped)."""
    if t > ts[-1]:
        return np.nan
    return float(np.interp(t, ts, mean))


def build_summary(run, ts, mean, train_steps, train_rew):
    best_i = int(np.argmax(mean))
    best = float(mean[best_i])
    # first time the run reaches 50% of its best eval reward
    idx50 = np.argmax(mean >= 0.5 * best)
    return {
        "env": run["env"],
        "run": run["run"],
        "lr": fmt_lr(run["lr"]),
        "seed": run["seed"],
        "n_steps": run["variant"].split("=")[1],
        "budget_steps": int(ts[-1]),
        "best_eval_reward": round(best, 1),
        "best_eval_step": int(ts[best_i]),
        "steps_to_50pct_best": int(ts[idx50]) if mean[idx50] >= 0.5 * best else np.nan,
        "eval_at_0.5M": round(eval_at(ts, mean, 0.5e6), 1),
        "eval_at_1M": round(eval_at(ts, mean, 1e6), 1),
        "eval_at_2M": round(eval_at(ts, mean, 2e6), 1),
        "final_train_ep_rew": round(float(np.mean(train_rew[-100:])), 1) if len(train_rew) else float("nan"),
        "sample_efficiency": round(float(np.mean(np.minimum(mean / best, 1.0))), 3),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _best_run_per_env(runs, eval_curves):
    """Highest best-eval-reward run for each environment."""
    best = {}
    for r in runs:
        b = float(np.max(eval_curves[r["run"]]["mean"]))
        if r["env"] not in best or b > best[r["env"]][0]:
            best[r["env"]] = (b, r)
    return {env: r for env, (_, r) in best.items()}


def fig_learning_curves(runs, eval_curves):
    """Fig 1 - 2x2 grid: eval reward vs steps, one panel per environment."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, env in zip(axes.flat, ENV_ORDER):
        env_runs = [r for r in runs if r["env"] == env]
        for i, r in enumerate(env_runs):
            c = eval_curves[r["run"]]
            col = run_color(r, i)
            ax.plot(c["ts"] / 1e6, c["mean"], label=label(r), lw=2.0, color=col,
                    linestyle="--" if r["variant"] != "n_steps=1024" else "-")
            ax.fill_between(c["ts"] / 1e6, c["mean"] - c["std"], c["mean"] + c["std"],
                            alpha=0.10, color=col)
        ax.set_title(f"{ENV_LABEL[env]}-v5  ({len(env_runs)} run{'s' if len(env_runs) > 1 else ''})")
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Eval reward (mean ± std, 10 eps)")
        ax.legend(loc="upper left")
    fig.suptitle("PPO learning curves - evaluation during training (deterministic policy)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_learning_curves.png"))
    plt.close(fig)


def fig_lr_comparison(runs, eval_curves):
    """Fig 2 - learning-rate comparison on Ant and Humanoid."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, env in zip(axes, ["Ant-v5", "Humanoid-v5"]):
        for i, r in enumerate(x for x in runs
                              if x["env"] == env and x["variant"] == "n_steps=1024"):
            c = eval_curves[r["run"]]
            ax.plot(c["ts"] / 1e6, c["mean"], color=run_color(r, i),
                    lw=2.0, label=f"lr {fmt_lr(r['lr'])}, seed {r['seed']}")
        ax.set_title(f"{ENV_LABEL[env]}-v5: lr 1e-4 vs 3e-4")
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Eval reward")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_lr_comparison.png"))
    plt.close(fig)


def fig_seed_robustness(runs, eval_curves):
    """Fig 3 - same hyperparameters, different seed (Humanoid lr1e-4)."""
    pair = [r for r in runs if r["env"] == "Humanoid-v5" and r["lr"] == "1e-04"]
    if len(pair) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in pair:
        c = eval_curves[r["run"]]
        ax.plot(c["ts"] / 1e6, c["mean"], lw=1.6, label=f"seed {r['seed']}")
    ax.set_title("Humanoid-v5: identical hyperparameters, different seed")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Eval reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_seed_robustness.png"))
    plt.close(fig)


def fig_nsteps(runs, eval_curves):
    """Fig 4 - Ant rollout-length ablation (1024 vs 2048, both lr 3e-4)."""
    pair = [r for r in runs if r["env"] == "Ant-v5" and r["lr"] == "3e-04"]
    if len(pair) < 2:
        return
    # explicit high-contrast A/B colors (the two palette entries for these runs
    # are both warm hues, too similar for a direct two-curve comparison)
    ab_colors = {"ant_ppo_lr3e-04": "#D55E00", "ant_ppo_2048_lr3e-04": "#0072B2"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in pair:
        c = eval_curves[r["run"]]
        ax.plot(c["ts"] / 1e6, c["mean"], lw=2.0, color=ab_colors.get(r["run"]),
                linestyle="--" if r["variant"] != "n_steps=1024" else "-",
                label=f"lr {fmt_lr(r['lr'])}, seed {r['seed']}, {r['variant']}")
    ax.set_title("Ant-v5: rollout length ablation (buffer 4×n_steps)")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Eval reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_nsteps_ablation.png"))
    plt.close(fig)


def fig_final_bars(final_evals):
    """Fig 5 - final 20-episode deterministic evaluation, all runs."""
    if not final_evals:
        return
    fig, ax = plt.subplots(figsize=(10, 4.6))
    order = sorted(final_evals, key=lambda r: (ENV_ORDER.index(r["env"]), -r["mean_reward"]))
    xs = np.arange(len(order))
    colors = [ENV_COLOR[r["env"]] for r in order]
    ax.bar(xs, [r["mean_reward"] for r in order], yerr=[r["std_reward"] for r in order],
           color=colors, alpha=0.85, capsize=3)
    best_per_env = {}
    for r in order:
        if r["env"] not in best_per_env or r["mean_reward"] > best_per_env[r["env"]]:
            best_per_env[r["env"]] = r["mean_reward"]
    for x, r in zip(xs, order):
        if r["mean_reward"] == best_per_env[r["env"]]:
            ax.text(x, r["mean_reward"] + r["std_reward"] + 60, "best", ha="center",
                    fontsize=8, fontweight="bold")
    def short_label(run_name):
        # Strip the redundant env-name prefix (already shown on line 1) so the
        # remaining tag (e.g. "lr1e-04_seed42") is short enough not to collide
        # with its neighbour at this figure width.
        tag = re.sub(r"^(hopper|walker2d|ant|humanoid)_ppo_?", "", run_name)
        return tag if tag else "(baseline)"

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{ENV_LABEL[r['env']]}\n{short_label(r['run'])}" for r in order],
                       fontsize=7.5, rotation=12, ha="right")
    ax.set_ylabel("Final eval reward (20 deterministic episodes)")
    ax.set_title("Final evaluation - best checkpoint of every run  (bold = best of its environment)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_final_evaluation.png"))
    plt.close(fig)


def fig_sample_efficiency(runs, eval_curves):
    """Fig 6 - normalized progress curves for the best run of each env."""
    best = _best_run_per_env(runs, eval_curves)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for env in ENV_ORDER:
        r = best[env]
        c = eval_curves[r["run"]]
        ax.plot(c["ts"] / 1e6, c["mean"] / np.max(c["mean"]), color=ENV_COLOR[env], lw=1.8,
                label=f"{ENV_LABEL[env]} ({int(c['ts'][-1]/1e6)}M budget)")
    ax.axhline(1.0, color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Eval reward / best eval reward")
    ax.set_title("Sample efficiency - how fast each task reaches its final skill")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_sample_efficiency.png"))
    plt.close(fig)


def fig_internals(runs, eval_curves, tb_data):
    """Fig 7 - PPO training dynamics: value fn, entropy, clipping, KL."""
    best = _best_run_per_env(runs, eval_curves)
    panels = [("train/explained_variance", "Explained variance (value function quality)"),
              ("train/entropy_loss", "Entropy loss (higher = more exploration)"),
              ("train/clip_fraction", "Clip fraction (updates hitting trust region)"),
              ("train/approx_kl", "Approximate KL (policy change per update)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (tag, title) in zip(axes.flat, panels):
        for env in ENV_ORDER:
            d = tb_data.get(best[env]["run"], {})
            if tag not in d:
                continue
            steps, vals = d[tag]
            frac = steps / steps[-1]
            k = max(1, len(vals) // 250)      # light decimation for plotting
            ax.plot(frac[::k], vals[::k], color=ENV_COLOR[env], lw=1.2, label=ENV_LABEL[env])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Fraction of training")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("PPO internals during training (best run per environment)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_ppo_internals.png"))
    plt.close(fig)


def fig_episode_length(runs, train_curves, eval_curves):
    """Fig 8 - training episode length (survival), best run per env."""
    best = _best_run_per_env(runs, eval_curves)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for env in ENV_ORDER:
        steps, _, lens = train_curves[best[env]["run"]]
        if len(lens) == 0:
            continue                      # run without Monitor CSVs (e.g. Hopper)
        smooth = rolling_mean(lens) if len(lens) >= 50 else lens
        x = steps[len(steps) - len(smooth):]        # align x with the valid-convolution output
        k = max(1, len(smooth) // 500)
        ax.plot(x[::k] / 1e6, smooth[::k], color=ENV_COLOR[env], lw=1.4, label=ENV_LABEL[env])
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Training episode length (rolling mean, 50 eps)")
    ax.set_title("Survival over training (Hopper/Walker2d/Ant terminate or cap at 1000 steps)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_episode_length.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary markdown
# --------------------------------------------------------------------------- #
def write_summary_md(summary, final_evals):
    finals = {f["run"]: f for f in final_evals}
    lines = [
        "# Run summary (auto-generated by analysis/analyze_ppo.py)",
        "",
        "All numbers come from the run artifacts: `evaluations.npz` (during-training eval,",
        "10 episodes/point) and `final_eval_results.json` (20 deterministic episodes on the",
        "best checkpoint, seed-2024 env). Reward is the raw environment reward.",
        "",
        "## Per-run table",
        "",
        "| Env | Run | LR | Seed | n_steps | Budget | Best eval | Final eval (20 eps) | Steps to 50% of best |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(summary, key=lambda x: (ENV_ORDER.index(x["env"]), -x["best_eval_reward"])):
        fe = finals.get(s["run"])
        final_txt = f"{fe['mean_reward']:.0f} ± {fe['std_reward']:.0f}" if fe else "—"
        s50 = s["steps_to_50pct_best"]
        s50_txt = f"{s50:.2e}" if s50 == s50 else "—"
        lines.append(
            f"| {ENV_LABEL[s['env']]} | `{s['run']}` | {s['lr']} | {s['seed']} | {s['n_steps']} "
            f"| {s['budget_steps']/1e6:.0f}M | {s['best_eval_reward']:.0f} | {final_txt} | {s50_txt} |"
        )
    lines += ["", "## Learning-rate comparison (1024-rollout runs only)", ""]
    for env in ["Ant-v5", "Humanoid-v5"]:
        rows = [s for s in summary if s["env"] == env and s["n_steps"] == "1024"]
        if len(rows) < 2:
            continue
        # one representative per lr (best seed if several seeds share an lr)
        by_lr = {}
        for s in rows:
            if s["lr"] not in by_lr or s["best_eval_reward"] > by_lr[s["lr"]]["best_eval_reward"]:
                by_lr[s["lr"]] = s
        a, b = sorted(by_lr.values(), key=lambda x: x["lr"])[:2]
        winner = a if a["best_eval_reward"] > b["best_eval_reward"] else b
        loser = b if winner is a else a
        lines.append(
            f"- **{ENV_LABEL[env]}**: lr {winner['lr']} reached {winner['best_eval_reward']:.0f} "
            f"vs {loser['best_eval_reward']:.0f} for lr {loser['lr']} "
            f"({winner['best_eval_reward'] - loser['best_eval_reward']:+.0f} in favour of lr {winner['lr']})."
        )
    lines += ["", "## Seed robustness (Humanoid, lr 1e-4)", ""]
    seeds = [s for s in summary if s["env"] == "Humanoid-v5" and s["lr"] == "1e-4"]
    if len(seeds) >= 2:
        vals = [s["best_eval_reward"] for s in seeds]
        lines.append(
            f"- Best eval across seeds: {min(vals):.0f} – {max(vals):.0f} "
            f"({(max(vals)-min(vals))/min(vals)*100:.0f}% spread) — single-seed results should be read cautiously."
        )
    lines += ["", "## Rollout-length ablation (Ant, lr 3e-4)", ""]
    ns = [s for s in summary if s["env"] == "Ant-v5" and s["lr"] == "3e-4"]
    if len(ns) >= 2:
        for s in ns:
            lines.append(f"- {s['n_steps']}: best eval {s['best_eval_reward']:.0f}, "
                         f"eval@1M {s['eval_at_1M']:.0f}, sample-efficiency score {s['sample_efficiency']:.2f}")
    out = os.path.join(RESULTS_DIR, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.relpath(out, ROOT)}")


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Discovering runs ...")
    runs = collect_runs()
    for r in runs:
        print(f"  {r['env']:12s} {r['run']:32s} (tb: {os.path.basename(r['event_dir'] or '-')})")

    eval_curves, train_curves, tb_data, summary = {}, {}, {}, []
    eval_rows, train_rows, tb_rows = [], [], []

    for r in runs:
        print(f"Loading {r['run']} ...")
        ts, mean, std, _ = load_eval_curve(r["log_dir"])
        eval_curves[r["run"]] = {"ts": ts, "mean": mean, "std": std}
        for t, m, s in zip(ts, mean, std):
            eval_rows.append({"env": r["env"], "run": r["run"], "timesteps": int(t),
                              "eval_mean": round(float(m), 2), "eval_std": round(float(s), 2)})

        steps, rew, lens = load_train_curve(r["log_dir"])
        train_curves[r["run"]] = (steps, rew, lens)
        for t, rw, ln in zip(steps, rew, lens):
            train_rows.append({"env": r["env"], "run": r["run"], "env_steps": int(t),
                               "ep_reward": round(float(rw), 2), "ep_length": int(ln)})

        if r["event_dir"]:
            tb_data[r["run"]] = load_tb_scalars(r["event_dir"])
            for tag, (tsteps, vals) in tb_data[r["run"]].items():
                for st, v in zip(tsteps, vals):
                    tb_rows.append({"env": r["env"], "run": r["run"], "tag": tag,
                                    "step": int(st), "value": float(v)})

        summary.append(build_summary(r, ts, mean, steps, rew))

    # final evaluations (best checkpoint, 20 deterministic episodes)
    final_evals = []
    for r in runs:
        fe = load_final_eval(r["model_dir"])
        if fe:
            fe["env"] = r["env"]
            fe["run"] = r["run"]          # some older JSONs lack run_name
            final_evals.append(fe)
        else:
            print(f"  WARNING: no final_eval_results.json for {r['run']}")

    print("Writing result tables ...")
    write_csv(os.path.join(RESULTS_DIR, "eval_curves.csv"), eval_rows,
              ["env", "run", "timesteps", "eval_mean", "eval_std"])
    write_csv(os.path.join(RESULTS_DIR, "train_curves.csv"), train_rows,
              ["env", "run", "env_steps", "ep_reward", "ep_length"])
    write_csv(os.path.join(RESULTS_DIR, "tb_scalars.csv"), tb_rows,
              ["env", "run", "tag", "step", "value"])
    write_csv(os.path.join(RESULTS_DIR, "runs_summary.csv"), summary,
              list(summary[0].keys()))
    write_csv(os.path.join(RESULTS_DIR, "final_evals.csv"),
              [{"env": f["env"], "run": f["run"], "checkpoint": f.get("checkpoint", "best_model"),
                "episodes": f["n_episodes"], "mean_reward": round(f["mean_reward"], 2),
                "std_reward": round(f["std_reward"], 2),
                "mean_length": round(f.get("mean_length", float("nan")), 1)}
               for f in final_evals],
              ["env", "run", "checkpoint", "episodes", "mean_reward", "std_reward", "mean_length"])
    write_summary_md(summary, final_evals)

    print("Generating figures ...")
    fig_learning_curves(runs, eval_curves)
    fig_lr_comparison(runs, eval_curves)
    fig_seed_robustness(runs, eval_curves)
    fig_nsteps(runs, eval_curves)
    fig_final_bars(final_evals)
    fig_sample_efficiency(runs, eval_curves)
    fig_internals(runs, eval_curves, tb_data)
    fig_episode_length(runs, train_curves, eval_curves)
    for f in sorted(os.listdir(FIGURES_DIR)):
        print(f"  figure: analysis/figures/{f}")
    print("Done.")


if __name__ == "__main__":
    main()




