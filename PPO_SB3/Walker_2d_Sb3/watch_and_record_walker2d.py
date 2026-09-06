"""
Watch a trained Walker2d PPO agent live in a MuJoCo window, or record it to .mp4.
Set MODE below - "human" and "rgb_array" rendering can't run from the same env
instance, so this is one or the other per run, not both at once.

Requires best_model.zip and vec_normalize.pkl from train_walker2d_ppo.py.
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

MODE = "watch"       # "watch" = live MuJoCo window | "record" = save an .mp4
N_EPISODES = 3        # used by "watch"
VIDEO_LENGTH = 1000    # steps, used by "record"

env_id = "Walker2d-v5"
model_dir = "./models/walker2d_ppo/"
video_dir = "./videos/walker2d_ppo/"


def load_env(render_mode):
    # render_mode must be set at env CREATION time for MuJoCo envs - setting it
    # afterward silently does nothing.
    env = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": render_mode})
    env = VecNormalize.load(f"{model_dir}/vec_normalize.pkl", env)
    env.training = False     # stop updating running normalization stats
    env.norm_reward = False  # not training, no need to normalize reward
    return env


def watch():
    env = load_env(render_mode="human")
    model = PPO.load(f"{model_dir}/best_model", env=env)

    for ep in range(N_EPISODES):
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


def record():
    os.makedirs(video_dir, exist_ok=True)
    env = load_env(render_mode="rgb_array")
    env = VecVideoRecorder(
        env, video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=VIDEO_LENGTH,
        name_prefix=f"{env_id}-agent",
    )
    model = PPO.load(f"{model_dir}/best_model", env=env)

    obs = env.reset()
    for _ in range(VIDEO_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    env.close()
    print(f"Video saved to {video_dir}")


if __name__ == "__main__":
    if MODE == "watch":
        watch()
    elif MODE == "record":
        record()
    else:
        raise ValueError("MODE must be 'watch' or 'record'")
