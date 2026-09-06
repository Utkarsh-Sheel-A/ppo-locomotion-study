"""
Generic final evaluation for any trained run in this repo.

Uses the exact same protocol as the per-environment evaluate_*.py scripts
(20 deterministic episodes, seed-2024 eval env, frozen VecNormalize stats,
raw/unnormalized rewards) so numbers are directly comparable everywhere.

Usage:
    python final_eval.py --model-dir ../Hopper_SB3/models/hopper_ppo --env-id Hopper-v5
"""
import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


def main(model_dir: str, env_id: str, checkpoint: str, n_eval_episodes: int):
    eval_env = make_vec_env(env_id, n_envs=1, seed=2024)
    eval_env = VecNormalize.load(os.path.join(model_dir, "vec_normalize.pkl"), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(os.path.join(model_dir, checkpoint), env=eval_env)

    episode_rewards, episode_lengths = [], []
    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        done, ep_reward, ep_length = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = eval_env.step(action)
            done = done_arr[0]
            ep_reward += reward[0]
            ep_length += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(f"Episode {ep + 1:2d}/{n_eval_episodes}: reward={ep_reward:8.2f}  length={ep_length:4d}")

    results = {
        "run_name": os.path.basename(model_dir.rstrip("/")),
        "env_id": env_id,
        "checkpoint": checkpoint,
        "n_episodes": n_eval_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }

    print("\n--- Final Evaluation ---")
    print(f"Run:          {results['run_name']}")
    print(f"Mean reward:  {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Mean length:  {results['mean_length']:.1f} +/- {results['std_length']:.1f}")

    out = os.path.join(model_dir, "final_eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--env-id", required=True)
    p.add_argument("--checkpoint", default="best_model", choices=["best_model", "final_model"])
    p.add_argument("--episodes", type=int, default=20)
    a = p.parse_args()
    main(a.model_dir, a.env_id, a.checkpoint, a.episodes)
