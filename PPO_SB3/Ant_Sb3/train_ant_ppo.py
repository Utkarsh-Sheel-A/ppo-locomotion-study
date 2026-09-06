"""
PPO on Ant-v5 with Stable-Baselines3. Same structure as the Walker2d script.

learning_rate, n_steps, and total_timesteps are CLI args so the lr comparison
and the rollout-length ablation are clean invocations of one script with a
fixed, matching budget - not hand-edited copies of the file that can drift
apart (this replaces what used to be a separate train_ant_ppo_2048.py).

    python train_ant_ppo.py --lr 3e-4                    # baseline
    python train_ant_ppo.py --lr 1e-4                    # lr comparison
    python train_ant_ppo.py --lr 3e-4 --n-steps 2048      # rollout-length ablation

Both runs log to a shared TensorBoard root, so `tensorboard --logdir ./tb_logs/ant_ppo/`
overlays them automatically for the comparison.
"""
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

USE_WANDB = False


def main(learning_rate: float, n_steps: int, total_timesteps: int):
    env_id = "Ant-v5"
    lr_tag = f"lr{learning_rate:.0e}"          # e.g. "lr3e-04" / "lr1e-04"
    # 1024 is the baseline and keeps the original run_name (matches existing
    # ant_ppo_lr3e-04 / ant_ppo_lr1e-04 runs); any other n_steps is tagged
    # explicitly, matching the existing ant_ppo_2048_lr3e-04 run's naming.
    run_name = f"ant_ppo_{lr_tag}" if n_steps == 1024 else f"ant_ppo_{n_steps}_{lr_tag}"

    log_dir = f"./logs/{run_name}/"
    model_dir = f"./models/{run_name}/"
    tb_root = "./tb_logs/ant_ppo/"             # SHARED across all runs, on purpose
    monitor_dir = os.path.join(log_dir, "monitor")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    print(f"--- Starting Training: {env_id} ({run_name}) ---")

    # UNCHANGED from Walker2d: same VecNormalize split, same reasoning.
    train_env = make_vec_env(env_id, n_envs=4, seed=42, monitor_dir=monitor_dir)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = make_vec_env(env_id, n_envs=1, seed=1337)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2500,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
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
            wandb.init(project="sb3-ant-ppo", name=run_name, sync_tensorboard=True, save_code=True)
            callbacks.append(WandbCallback())
        except ImportError:
            print("wandb not installed - continuing without it.")

    # Every value below is explicit rather than left at SB3 defaults. The point
    # of these runs is "only one thing changed at a time" - so everything else
    # needs to be visibly pinned in the code, not silently defaulted.
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=n_steps,                # <-- varied for the rollout-length ablation
        batch_size=256,
        learning_rate=learning_rate,    # <-- varied for the lr comparison
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        seed=42,
        verbose=1,
        tensorboard_log=tb_root,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=run_name.replace("ant_ppo_", ""),  # keeps tb subfolder names short but distinct
    )

    model.save(f"{model_dir}/final_model")
    train_env.save(f"{model_dir}/vec_normalize.pkl")
    print(f"--- Training Complete: {env_id} ({run_name}) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate, e.g. 3e-4 or 1e-4")
    parser.add_argument("--n-steps", type=int, default=1024,
                         help="rollout length per env; 2048 fixed Ant's mid-training plateau (see cheatsheet)")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                         help="keep this IDENTICAL across runs being compared")
    args = parser.parse_args()
    main(learning_rate=args.lr, n_steps=args.n_steps, total_timesteps=args.timesteps)
