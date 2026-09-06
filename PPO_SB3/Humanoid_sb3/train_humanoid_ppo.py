"""
PPO on Humanoid-v5 with Stable-Baselines3. Same structure as the Ant/Walker2d scripts.

Three-iteration plan (lr and seed via CLI, everything else pinned and identical
across all three per your instruction not to sweep further):

    # Iteration 1 - baseline
    python train_humanoid_ppo.py --lr 3e-4 --seed 42

    # Iteration 2 - lr only, changed
    python train_humanoid_ppo.py --lr 1e-4 --seed 42

    # Iteration 3 - final portfolio run: rerun whichever of 1/2 won, new seed
    # (same hyperparameters, different seed - checks the win wasn't just that seed)
    python train_humanoid_ppo.py --lr <winning_lr> --seed 7

"Clearly fails to learn" (the only condition under which to deviate from this
plan, per your instruction) means, operationally:
  - eval/mean_reward is still flat near its early floor past ~2.5M of the 5M
    budget, no upward trend at all, OR
  - entropy_loss collapses to near-zero in the first few hundred k steps while
    reward is still flat (policy went deterministic before finding anything), OR
  - explained_variance stays near zero/negative for most of the run (value
    function never gets traction - usually points at something more basic
    being wrong than a hyperparameter choice).
If that happens, the fallback order (not applied by default here) is:
  1. policy_kwargs=dict(net_arch=[256, 256]) - default net may be too small for
     Humanoid's ~3x larger action space vs Ant
  2. ent_coef > 0 (e.g. 0.005-0.01) - standing-still-and-not-falling is a much
     richer reward source here than on Ant, easier local optimum to get stuck in
  3. n_steps=2048 - what actually fixed Ant's plateau last time
"""
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

USE_WANDB = False


def main(learning_rate: float, seed: int, total_timesteps: int):
    env_id = "Humanoid-v5"
    lr_tag = f"lr{learning_rate:.0e}"
    run_name = f"humanoid_ppo_{lr_tag}_seed{seed}"

    log_dir = f"./logs/{run_name}/"
    model_dir = f"./models/{run_name}/"
    tb_root = "./tb_logs/humanoid_ppo/"   # shared root - overlays all iterations in one TensorBoard view
    monitor_dir = os.path.join(log_dir, "monitor")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    print(f"--- Starting Training: {env_id} ({lr_tag}, seed={seed}) ---")

    # UNCHANGED from Ant/Walker2d: same VecNormalize split, same n_envs=4.
    train_env = make_vec_env(env_id, n_envs=4, seed=seed, monitor_dir=monitor_dir)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = make_vec_env(env_id, n_envs=1, seed=1337)  # fixed across all iterations on purpose
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
        save_freq=200_000 // 4,   # every ~200k env transitions - Humanoid runs are longer, checkpoint less often
        save_path=model_dir,
        name_prefix="ckpt",
    )
    callbacks = [eval_callback, checkpoint_callback]

    if USE_WANDB:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(project="sb3-humanoid-ppo", name=run_name, sync_tensorboard=True, save_code=True)
            callbacks.append(WandbCallback())
        except ImportError:
            print("wandb not installed - continuing without it.")

    # Every value pinned explicitly and IDENTICAL across all 3 iterations except
    # learning_rate and seed, per your instruction. No architecture or other
    # hyperparameter changes unless the "fails to learn" criteria above trigger.
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,
        batch_size=256,
        learning_rate=learning_rate,   # <-- varied across iterations 1 and 2
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,                     # <-- varied only for iteration 3
        verbose=1,
        tensorboard_log=tb_root,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=f"{lr_tag}_seed{seed}",
    )

    model.save(f"{model_dir}/final_model")
    train_env.save(f"{model_dir}/vec_normalize.pkl")
    print(f"--- Training Complete: {env_id} ({lr_tag}, seed={seed}) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                         help="keep IDENTICAL across all 3 iterations")
    args = parser.parse_args()
    main(learning_rate=args.lr, seed=args.seed, total_timesteps=args.timesteps)
