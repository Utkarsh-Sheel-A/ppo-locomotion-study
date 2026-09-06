"""
Records a short video of a trained PPO agent.
Run this AFTER train_hopper_ppo.py has produced best_model.zip and vec_normalize.pkl.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

env_id = "Hopper-v5"
model_dir = "./models/hopper_ppo/"
video_dir = "./videos/hopper_ppo/"

# render_mode must be set at env CREATION time for MuJoCo envs - setting it
# afterward silently does nothing, and VecVideoRecorder will fail or record blank frames.
video_env = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": "rgb_array"})
video_env = VecNormalize.load(f"{model_dir}/vec_normalize.pkl", video_env)
video_env.training = False     # stop updating the running normalization stats
video_env.norm_reward = False  # we're not training, so no need to normalize reward

video_env = VecVideoRecorder(
    video_env, video_dir,
    record_video_trigger=lambda step: step == 0,
    video_length=1000,
    name_prefix=f"{env_id}-agent",
)

model = PPO.load(f"{model_dir}/best_model", env=video_env)
obs = video_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = video_env.step(action)

video_env.close()
print("Video saved!")
