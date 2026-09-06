"""
PPO on Hopper-v5 with Stable-Baselines3.
Change env_id below to scale up to Ant-v5 later - nothing else needs to change.
See ppo_sb3_cheatsheet.md for what each hyperparameter/metric actually means.
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback


def main():
    env_id = "Hopper-v5"
    run_name = "hopper_ppo"

    log_dir = f"./logs/{run_name}/"
    model_dir = f"./models/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"--- Starting Training: {env_id} ---")

    # Train env: normalize obs AND reward - this is what keeps PPO's updates stable
    # on continuous-control tasks where raw reward scale can vary a lot.
    train_env = make_vec_env(env_id, n_envs=4, seed=42)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Eval env: normalize obs the same way, but NEVER normalize reward here -
    # you want the real, comparable score, not a training-stabilized one.
    # EvalCallback automatically copies obs stats from train_env before each eval.
    eval_env = make_vec_env(env_id, n_envs=1, seed=1337)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2500,      # 10,000 env steps total (4 envs * 2500)
        n_eval_episodes=10,  # default is 5 - too noisy for Hopper on its own
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,        # rollout length per env -> buffer size = 1024*4 = 4096
        batch_size=256,      # divides the buffer evenly -> 16 minibatches/epoch
        learning_rate=3e-4,
        seed=42,             # reproducibility - seeds the model too, not just the envs
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    model.save(f"{model_dir}/final_model")
    train_env.save(f"{model_dir}/vec_normalize.pkl")
    print(f"--- Training Complete: {env_id} ---")
    print(f"Run record_video.py to watch the trained agent.")


if __name__ == "__main__":
    main()
