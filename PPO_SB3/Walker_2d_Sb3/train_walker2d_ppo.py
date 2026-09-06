"""
PPO on Walker2d-v5 with Stable-Baselines3.
Structure ported from the Hopper training script - see the CHANGE comments below
for exactly what differs and why.

Set USE_WANDB = True below if you want W&B logging (requires `pip install wandb`
and being logged in). Off by default so this runs with zero extra setup.
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

USE_WANDB = False  # flip to True once wandb is installed + configured


def main():
    # CHANGE (required - Walker2d-specific): env_id swapped. Nothing else about
    # make_vec_env needs to change - Walker2d is registered the same way as Hopper.
    env_id = "Walker2d-v5"
    run_name = "walker2d_ppo"

    log_dir = f"./logs/{run_name}/"
    model_dir = f"./models/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"--- Starting Training: {env_id} ---")

    # ADDED (infra gap-fill, not a Hopper->Walker2d change): monitor_dir writes
    # per-episode reward/length to CSV so plot_results.py can chart the training
    # curve without parsing TensorBoard event files.
    monitor_dir = os.path.join(log_dir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    # UNCHANGED: same train/eval VecNormalize split as Hopper. Reward scale
    # differs between the two tasks, but the *mechanism* for handling it doesn't.
    train_env = make_vec_env(env_id, n_envs=4, seed=42, monitor_dir=monitor_dir)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = make_vec_env(env_id, n_envs=1, seed=1337)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # UNCHANGED: same eval cadence and episode count as Hopper.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2500,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # ADDED (infra gap-fill): periodic checkpoints, since you said this existed
    # in your Hopper setup. save_freq is per-env steps, so this saves roughly
    # every 100k env transitions with n_envs=4.
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // 4,
        save_path=model_dir,
        name_prefix="ckpt",
    )

    callbacks = [eval_callback, checkpoint_callback]

    if USE_WANDB:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback

            wandb.init(
                project="sb3-walker2d-ppo",
                name=run_name,
                sync_tensorboard=True,   # reuses the same TB logs, no duplicate logging code
                monitor_gym=False,
                save_code=True,
            )
            callbacks.append(WandbCallback())
        except ImportError:
            print("wandb not installed - continuing without it. `pip install wandb` to enable.")

    # UNCHANGED: same PPO hyperparameters as Hopper, per your instruction not to
    # retune from scratch. Only total_timesteps below is bumped - see the
    # "training budget" section in chat for why, not baked in as a "required" change.
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        seed=42,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # CHANGE (recommended, not required): 2M vs Hopper's 1M. Walker2d's larger
    # action space and bipedal coordination problem generally need more experience
    # to get past the "hop / shuffle" local optima than Hopper's single-leg task does.
    model.learn(total_timesteps=2_000_000, callback=CallbackList(callbacks))

    model.save(f"{model_dir}/final_model")
    train_env.save(f"{model_dir}/vec_normalize.pkl")
    print(f"--- Training Complete: {env_id} ---")
    print(f"Run evaluate_walker2d.py for a final evaluation, or plot_results.py to chart the run.")

    # To resume training later from a checkpoint (same pattern applies to Hopper):
    #   model = PPO.load(f"{model_dir}/ckpt_<steps>_steps.zip", env=train_env)
    #   model.learn(total_timesteps=<more>, reset_num_timesteps=False, callback=...)
    # Remember: any saved model is only meaningful paired with the vec_normalize.pkl
    # from the SAME run - don't mix normalization stats across runs.


if __name__ == "__main__":
    main()
