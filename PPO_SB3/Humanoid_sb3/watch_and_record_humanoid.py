"""
Watch a trained Humanoid PPO agent live in a MuJoCo window, or record it to .mp4.
Picks the run by --lr/--seed, matching train_humanoid_ppo.py's naming
(humanoid_ppo_lr<value>_seed<value>).

    python watch_and_record_humanoid.py --lr 3e-4 --seed 42                    # live window
    python watch_and_record_humanoid.py --lr 3e-4 --seed 42 --mode record       # save an .mp4

"human" and "rgb_array" rendering can't run from the same env instance, so
this is one or the other per invocation, not both at once.
"""
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

env_id = "Humanoid-v5"


def load_env(model_dir, render_mode):
    # render_mode must be set at env CREATION time for MuJoCo envs - setting it
    # afterward silently does nothing.
    env = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": render_mode})
    env = VecNormalize.load(f"{model_dir}/vec_normalize.pkl", env)
    env.training = False     # stop updating running normalization stats
    env.norm_reward = False  # not training, no need to normalize reward
    return env


def watch(model_dir, n_episodes):
    env = load_env(model_dir, render_mode="human")
    model = PPO.load(f"{model_dir}/best_model", env=env)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            done = done_arr[0]
            ep_reward += reward[0]
            env.render()  # explicit call - harmless even if "human" mode already auto-renders
        print(f"Episode {ep + 1}: reward={ep_reward:.2f}")

    env.close()


def record(model_dir, video_dir, video_length):
    os.makedirs(video_dir, exist_ok=True)
    env = load_env(model_dir, render_mode="rgb_array")
    env = VecVideoRecorder(
        env, video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=f"{env_id}-agent",
    )
    model = PPO.load(f"{model_dir}/best_model", env=env)

    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    env.close()
    print(f"Video saved to {video_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="which run to load, e.g. 3e-4 or 1e-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["watch", "record"], default="watch")
    parser.add_argument("--episodes", type=int, default=3, help="used by watch mode")
    parser.add_argument("--video-length", type=int, default=1000, help="steps, used by record mode")
    args = parser.parse_args()

    lr_tag = f"lr{args.lr:.0e}"
    run_name = f"humanoid_ppo_{lr_tag}_seed{args.seed}"
    model_dir = f"./models/{run_name}/"
    video_dir = f"./videos/{run_name}/"

    if args.mode == "watch":
        watch(model_dir, args.episodes)
    else:
        record(model_dir, video_dir, args.video_length)
