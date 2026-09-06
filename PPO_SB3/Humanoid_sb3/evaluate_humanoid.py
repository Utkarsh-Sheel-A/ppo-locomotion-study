"""
Standalone final evaluation for a trained Humanoid PPO model.
Picks the run by --lr/--seed, matching train_humanoid_ppo.py's naming
(humanoid_ppo_lr<value>_seed<value>).

    python evaluate_humanoid.py --lr 3e-4 --seed 42
    python evaluate_humanoid.py --lr 1e-4 --seed 42
    python evaluate_humanoid.py --lr <winner> --seed 7   # iteration 3

Reports: mean reward, std reward, mean/std episode length, over N deterministic episodes.
"""
import argparse
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

env_id = "Humanoid-v5"
n_eval_episodes = 20  # more than training-time eval (10) since this is the number you'll report


def main(learning_rate: float, seed: int, checkpoint: str):
    lr_tag = f"lr{learning_rate:.0e}"
    run_name = f"humanoid_ppo_{lr_tag}_seed{seed}"
    model_dir = f"./models/{run_name}/"

    eval_env = make_vec_env(env_id, n_envs=1, seed=2024)
    eval_env = VecNormalize.load(f"{model_dir}/vec_normalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(f"{model_dir}/{checkpoint}", env=eval_env)

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = eval_env.step(action)
            done = done_arr[0]
            ep_reward += reward[0]
            ep_length += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(f"Episode {ep + 1:2d}/{n_eval_episodes}: reward={ep_reward:8.2f}  length={ep_length:4d}")

    results = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "n_episodes": n_eval_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }

    print("\n--- Final Evaluation ---")
    print(f"Run:          {run_name}")
    print(f"Checkpoint:   {checkpoint}")
    print(f"Mean reward:  {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Mean length:  {results['mean_length']:.1f} +/- {results['std_length']:.1f}")

    with open(f"{model_dir}/final_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {model_dir}/final_eval_results.json")

    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="which run to evaluate, e.g. 3e-4 or 1e-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="best_model", choices=["best_model", "final_model"])
    args = parser.parse_args()
    main(learning_rate=args.lr, seed=args.seed, checkpoint=args.checkpoint)
