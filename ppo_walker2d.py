#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# ---------------------------
# Utils
# ---------------------------

def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)

class RMSNormObs:
    """Running mean/var normalizer for observations (per-dim)."""
    def __init__(self, shape, eps=1e-8, clip=10.0, device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)  # avoid div by 0
        self.eps = eps
        self.clip = clip
        self.device = device

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # x: [B, obs_dim]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = torch.clamp(new_var, min=1e-6)
        self.count = tot_count

    @torch.no_grad()
    def normalize(self, x: torch.Tensor):
        x = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(x, -self.clip, self.clip)

def _wrap_record_episode_statistics(env, size: int = 1000):
    """Compatibility shim for RecordEpisodeStatistics across gym/gymnasium versions.

    Tries parameter names in order: buffer_length (gymnasium>=1.2), buffer_size, deque_size (older).
    Falls back to default constructor if none are accepted.
    """
    try:
        return gym.wrappers.RecordEpisodeStatistics(env, buffer_length=size)
    except TypeError:
        try:
            return gym.wrappers.RecordEpisodeStatistics(env, buffer_size=size)
        except TypeError:
            try:
                return gym.wrappers.RecordEpisodeStatistics(env, deque_size=size)
            except TypeError:
                return gym.wrappers.RecordEpisodeStatistics(env)

# ---------------------------
# Model
# ---------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        # Policy
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        # Value
        self.v  = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # State-independent log-std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Init
        for m in self.pi:
            if isinstance(m, nn.Linear):
                gain = np.sqrt(2)
                if m is self.pi[-1]:
                    gain = 0.01
                orthogonal_init(m, gain)
        for m in self.v:
            if isinstance(m, nn.Linear):
                gain = np.sqrt(2)
                if m is self.v[-1]:
                    gain = 1.0
                orthogonal_init(m, gain)

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mu = self.pi(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)
        value = self.v(obs).squeeze(-1)
        return dist, value

# ---------------------------
# Storage (rollout buffer)
# ---------------------------

@dataclass
class PPOArgs:
    env_id: str = "Walker2d-v5"
    seed: int = 1
    total_steps: int = 10_000_000
    num_envs: int = 16
    num_steps: int = 1024             # per update per env
    minibatch_size: int = 1024
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    target_kl: float = 0.015          # early stop per update if exceeded
    eval_interval_updates: int = 10
    eval_episodes: int = 10
    solve_threshold: float = 4000.0   # (opinion) define "solved" here

class Rollout:
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        self.obs     = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float32, device=device)
        self.logp    = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rew     = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.done    = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.val     = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.adv     = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.ret     = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.step_idx = 0

    def add(self, obs, actions, logp, rew, done, val):
        t = self.step_idx
        self.obs[t] = obs
        self.actions[t] = actions
        self.logp[t] = logp
        self.rew[t] = rew
        self.done[t] = done
        self.val[t] = val
        self.step_idx += 1

    def compute_gae(self, last_val, gamma, lam):
        T = self.rew.shape[0]
        adv = torch.zeros_like(self.rew)
        lastgaelam = torch.zeros_like(self.rew[0])
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - self.done[t]
            next_values = last_val if t == T - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * next_values * next_nonterminal - self.val[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            adv[t] = lastgaelam
        self.adv = adv
        self.ret = self.adv + self.val

    def get_minibatches(self, mb_size, shuffle=True):
        T, N = self.rew.shape
        total = T * N
        obs = self.obs.reshape(total, -1)
        actions = self.actions.reshape(total, -1)
        logp = self.logp.reshape(total)
        adv = self.adv.reshape(total)
        ret = self.ret.reshape(total)
        val = self.val.reshape(total)

        idxs = torch.arange(total, device=obs.device)
        if shuffle:
            idxs = idxs[torch.randperm(total, device=obs.device)]

        for start in range(0, total, mb_size):
            end = start + mb_size
            mb_idxs = idxs[start:end]
            yield obs[mb_idxs], actions[mb_idxs], logp[mb_idxs], adv[mb_idxs], ret[mb_idxs], val[mb_idxs]

# ---------------------------
# Env helpers
# ---------------------------

def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env = _wrap_record_episode_statistics(env, size=1000)
        env = gym.wrappers.ClipAction(env)  # enforce bounds [-1,1]
        return env
    return thunk

def eval_policy(env_id, policy: ActorCritic, obs_norm: RMSNormObs, episodes=10, seed=10_000, device="cpu"):
    env = gym.make(env_id)
    env = gym.wrappers.ClipAction(env)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                obs_n = obs_norm.normalize(obs_t)
                dist, _ = policy(obs_n)
                action = dist.mean  # deterministic (mean action) for eval
            action = action.squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(r)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))

def load_trained_policy(checkpoint_path: str, obs_dim: int, act_dim: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    policy = ActorCritic(obs_dim, act_dim, hidden=256).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    obs_norm = RMSNormObs(obs_dim, device=device)
    stats = ckpt.get("obs_norm")
    if stats is not None:
        if "mean" in stats:
            obs_norm.mean.copy_(stats["mean"].to(device))
        if "var" in stats:
            obs_norm.var.copy_(stats["var"].to(device))
        if "count" in stats:
            obs_norm.count = stats["count"].to(device)
    return policy, obs_norm

def run_saved_policy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = {}
    if args.render_mode:
        env_kwargs["render_mode"] = args.render_mode
    env = gym.make(args.env_id, **env_kwargs)
    env = gym.wrappers.ClipAction(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy, obs_norm = load_trained_policy(args.checkpoint, obs_dim, act_dim, device)

    returns = []
    for ep in range(args.play_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_n = obs_norm.normalize(obs_t)
            with torch.no_grad():
                dist, _ = policy(obs_n)
                action = dist.sample() if args.sample_actions else dist.mean
            action = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)
        print(f"Episode {ep + 1}: return={ep_ret:.1f}")
    env.close()

    if returns:
        mean = float(np.mean(returns))
        std = float(np.std(returns))
        print(f"Average return over {len(returns)} episodes: {mean:.1f} ± {std:.1f}")

# ---------------------------
# Training
# ---------------------------

def train(args: PPOArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env setup
    env_fns = [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    obs, _ = envs.reset(seed=args.seed)

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    # Policy
    ac = ActorCritic(obs_dim, act_dim, hidden=256).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)

    obs_norm = RMSNormObs(obs_dim, device=device)

    num_updates = args.total_steps // (args.num_steps * args.num_envs)

    # Logging helpers
    ep_returns = []
    best_eval = -np.inf
    start_time = time.time()

    for update in range(1, num_updates + 1):
        rollout = Rollout(args.num_steps, args.num_envs, obs_dim, act_dim, device)

        for t in range(args.num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                obs_norm.update(obs_t)  # track running stats online
                obs_n = obs_norm.normalize(obs_t)
                dist, val = ac(obs_n)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)

            actions_np = action.cpu().numpy()
            next_obs, rew, term, trunc, infos = envs.step(actions_np)
            done = np.logical_or(term, trunc).astype(np.float32)

            # Episode stats from vector envs (Gymnasium puts them in 'final_info')
            if "final_info" in infos and infos["final_info"] is not None:
                for finfo in infos["final_info"]:
                    if finfo and "episode" in finfo:
                        ep_returns.append(finfo["episode"]["r"])

            rollout.add(obs_t, action, logp, torch.as_tensor(rew, device=device, dtype=torch.float32),
                        torch.as_tensor(done, device=device, dtype=torch.float32), val)

            obs = next_obs

        # Bootstrap last value
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_n = obs_norm.normalize(obs_t)
            _, last_val = ac(obs_n)

        rollout.compute_gae(last_val, args.gamma, args.gae_lambda)

        # Normalize advantages
        adv_mean, adv_std = rollout.adv.mean(), rollout.adv.std(unbiased=False) + 1e-8
        rollout.adv = (rollout.adv - adv_mean) / adv_std

        # PPO update
        b_obs = rollout.obs.reshape(-1, obs_dim)
        approx_kl = 0.0
        clipfrac_list = []

        for epoch in range(args.update_epochs):
            for mb in rollout.get_minibatches(args.minibatch_size, shuffle=True):
                mb_obs, mb_actions, mb_logp_old, mb_adv, mb_ret, _ = mb
                obs_n = obs_norm.normalize(mb_obs)

                dist, v = ac(obs_n)
                logp = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (logp - mb_logp_old).exp()
                with torch.no_grad():
                    approx_kl = (mb_logp_old - logp).mean().item()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                clipfrac_list.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                # Value loss with clipping
                v_pred_clipped = rollout.val.reshape(-1)[0:0]  # placeholder to keep shapes explicit
                # Properly compute value clipping using old values from buffer for the same indices
                # We reconstruct old values aligned with mb indices:
                # (A tiny overhead, but clarity > micro-optim)
                with torch.no_grad():
                    # old values for the same mb indices
                    old_v = rollout.val.reshape(-1)[mb_logp_old.shape[0]*0 : ]  # dummy slice to enable indexing
                # The above trick doesn't give us correct old_v directly; instead, recompute cleanly:
                # Simpler: don't use v-clip (works fine). If you want v-clip, keep old values in the minibatch too.
                value_loss = 0.5 * (mb_ret - v).pow(2).mean()

                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad_norm)
                optimizer.step()

            if approx_kl > args.target_kl:
                break  # early stop this update

        # Logging
        fps = int((args.num_envs * args.num_steps) / (time.time() - start_time + 1e-9))
        start_time = time.time()
        mean_return = float(np.mean(ep_returns[-10:])) if len(ep_returns) >= 1 else float("nan")
        clipfrac = np.mean(clipfrac_list) if clipfrac_list else 0.0
        print(f"update={update:04d}  steps={(update*args.num_envs*args.num_steps):,}  "
              f"avg_return_10ep={mean_return:.1f}  kl={approx_kl:.4f}  clipfrac={clipfrac:.2f}  fps~{fps}")

        # Periodic eval
        if update % args.eval_interval_updates == 0:
            eval_mean, eval_std = eval_policy(args.env_id, ac, obs_norm,
                                              episodes=args.eval_episodes, seed=10000, device=device)
            print(f"[eval] mean={eval_mean:.1f} ± {eval_std:.1f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save({"model": ac.state_dict(),
                            "obs_norm": {"mean": obs_norm.mean, "var": obs_norm.var, "count": obs_norm.count}},
                           "ppo_walker2d_best.pt")
            if eval_mean >= args.solve_threshold:
                print(f"Reached solve threshold ({args.solve_threshold}). Saved model and exiting.")
                break

    envs.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=PPOArgs.env_id)
    p.add_argument("--seed", type=int, default=PPOArgs.seed)
    p.add_argument("--total-steps", type=int, default=PPOArgs.total_steps)
    p.add_argument("--num-envs", type=int, default=PPOArgs.num_envs)
    p.add_argument("--num-steps", type=int, default=PPOArgs.num_steps)
    p.add_argument("--minibatch-size", type=int, default=PPOArgs.minibatch_size)
    p.add_argument("--update-epochs", type=int, default=PPOArgs.update_epochs)
    p.add_argument("--gamma", type=float, default=PPOArgs.gamma)
    p.add_argument("--gae-lambda", type=float, default=PPOArgs.gae_lambda)
    p.add_argument("--clip-coef", type=float, default=PPOArgs.clip_coef)
    p.add_argument("--ent-coef", type=float, default=PPOArgs.ent_coef)
    p.add_argument("--vf-coef", type=float, default=PPOArgs.vf_coef)
    p.add_argument("--max-grad-norm", type=float, default=PPOArgs.max_grad_norm)
    p.add_argument("--lr", type=float, default=PPOArgs.lr)
    p.add_argument("--target-kl", type=float, default=PPOArgs.target_kl)
    p.add_argument("--eval-interval-updates", type=int, default=PPOArgs.eval_interval_updates)
    p.add_argument("--eval-episodes", type=int, default=PPOArgs.eval_episodes)
    p.add_argument("--solve-threshold", type=float, default=PPOArgs.solve_threshold)
    p.add_argument("--run-saved", action="store_true", help="Run a saved policy instead of training")
    p.add_argument("--checkpoint", type=str, default="ppo_walker2d_best.pt")
    p.add_argument("--play-episodes", type=int, default=5)
    p.add_argument("--render-mode", type=str, default=None)
    p.add_argument("--sample-actions", action="store_true", help="Sample stochastic actions when running a saved policy")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    if a.run_saved:
        run_saved_policy(a)
    else:
        train(PPOArgs(
            env_id=a.env_id,
            seed=a.seed,
            total_steps=a.total_steps,
            num_envs=a.num_envs,
            num_steps=a.num_steps,
            minibatch_size=a.minibatch_size,
            update_epochs=a.update_epochs,
            gamma=a.gamma,
            gae_lambda=a.gae_lambda,
            clip_coef=a.clip_coef,
            ent_coef=a.ent_coef,
            vf_coef=a.vf_coef,
            max_grad_norm=a.max_grad_norm,
            lr=a.lr,
            target_kl=a.target_kl,
            eval_interval_updates=a.eval_interval_updates,
            eval_episodes=a.eval_episodes,
            solve_threshold=a.solve_threshold,
        ))
