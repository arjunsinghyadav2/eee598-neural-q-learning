"""
Episodic stochastic shortest-path MDP: neural Q-learning vs. tabular Q-learning
EEE598 Reinforcement Learning, Assignment 3
Team 9: Chinmay Prashanth, Arjun Singh Yadav, Muhammad Ibraheem

The Q table of Assignment 2 is replaced by a feedforward network trained only
from sampled transitions, with experience replay and a hard-copied target
network. Costs stay positive and are minimized, gamma = 1, V(h) = 0.

Run:
    python neural_q_learning.py                 self-checks (abort on failure), seeds 0-4,
                                                tabular baselines, evaluation, figures, tables
    python neural_q_learning.py --checks-only   self-checks only
    python neural_q_learning.py --plots-only    rebuild the figures from the saved logs
    python neural_q_learning.py --sweep         sensitivity study on held-out seeds 100, 101, ...

Options:
    --seeds 0 1 2 3 4   seeds of the main run (these are the defaults)
    --episodes N        training length; if N is not the default 5000 the epsilon decay becomes
                        round(0.8 * N) and the results are not the official ones
    --out DIR           output folder (default: results/ next to this file)
    --skip-checks       do not run the self-checks first
    --jobs N            worker processes of the sweep (default: half of the CPU count)
    --sweep-seeds K     number of held-out seeds of the sweep (default: 10)

Produces (written to results/ next to this file, or to --out DIR; a main run first deletes the
logs, tables and figures that an earlier main run left there):
    checks.txt               - one PASS or FAIL line per self-check
    results.json             - meta, config, protocol, reference (V*, Q*, pi*), one record per
                               method (neural, tabular_a2, tabular_matched) and seed, and the
                               mean and standard deviation of each method across the seeds
    five_seed_evaluation.csv - 1000-episode evaluation statistics per method and seed
    policies.csv             - greedy actions, nominal path and exact expected cost
    training_log_seed{n}.npz - neural logs: episode_costs, epsilons, losses, q_hist
    tabular_log_seed{n}.npz  - tabular logs of both baselines, keys prefixed a2_ and matched_
    fig_seed4_cost, fig_seed4_loss, fig_stability, fig_q_rmse - each as .pdf and .png
    sweep_runs.csv, sweep_summary.csv - only with --sweep
Every q_hist has one row per episode and these 10 columns:
    a|Ab, a|Ac, b|Ad, b|Ae, c|Ae, c|Af, d|Ag, e|Ag, f|Ag, g|Ah

Software: Python 3.14, torch 2.10 (CPU), numpy 2.4, matplotlib 3.10.
"""

import argparse
import concurrent.futures
import copy
import csv
import dataclasses
import hashlib
import inspect
import json
import multiprocessing
import os
import platform
import random
import re
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolved from this file so that the working directory never decides where results go.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --------------------------------------------------------------------------
# 1. Environment definition (tables copied verbatim from Assignment 2)
# --------------------------------------------------------------------------
# Branching states a, b, c each have two actions. At each branching state the
# intended transition succeeds w.p. 0.8 and the alternative branch is taken
# w.p. 0.2. States d, e, f, g have a single action (no decision, deterministic
# successor). h is terminal, V(h) = 0. The cost incurred on a transition is the
# edge cost of the branch that is ACTUALLY realized (not the intended one), a
# positive number to minimize. There is no reward sign flip anywhere.

ACTIONS = {
    "a": ["Ab", "Ac"],
    "b": ["Ad", "Ae"],
    "c": ["Ae", "Af"],
    "d": ["Ag"],
    "e": ["Ag"],
    "f": ["Ag"],
    "g": ["Ah"],
}

# transition[s][a] = list of (next_state, prob, cost)
TRANSITIONS = {
    "a": {
        "Ab": [("b", 0.8, 2), ("c", 0.2, 2)],
        "Ac": [("c", 0.8, 2), ("b", 0.2, 2)],
    },
    "b": {
        "Ad": [("d", 0.8, 3), ("e", 0.2, 2)],
        "Ae": [("e", 0.8, 2), ("d", 0.2, 3)],
    },
    "c": {
        "Ae": [("e", 0.8, 5), ("f", 0.2, 4)],
        "Af": [("f", 0.8, 4), ("e", 0.2, 5)],
    },
    "d": {"Ag": [("g", 1.0, 3)]},
    "e": {"Ag": [("g", 1.0, 3)]},
    "f": {"Ag": [("g", 1.0, 2)]},
    "g": {"Ah": [("h", 1.0, 1)]},
}

STATES = ["a", "b", "c", "d", "e", "f", "g", "h"]
TERMINAL = "h"
START = "a"
BRANCHING_STATES = ["a", "b", "c"]

# Fixed order of the network outputs. A label names its intended successor: "Ab" intends b.
ACTION_LABELS = ["Ab", "Ac", "Ad", "Ae", "Af", "Ag", "Ah"]
N_STATES = len(STATES)
N_ACTIONS = len(ACTION_LABELS)
STATE_INDEX = {s: i for i, s in enumerate(STATES)}
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_LABELS)}

# The 10 feasible (state, action) pairs, in the column order of every q_hist array:
# a|Ab, a|Ac, b|Ad, b|Ae, c|Ae, c|Af, d|Ag, e|Ag, f|Ag, g|Ah.
FEASIBLE_PAIRS = [(s, a) for s in STATES if s != TERMINAL for a in ACTIONS[s]]


def step(rng, s, a):
    """Sample a transition (s,a) -> (s', cost)."""
    outcomes = TRANSITIONS[s][a]
    probs = [o[1] for o in outcomes]
    idx = rng.choice(len(outcomes), p=probs)
    s_next, _, cost = outcomes[idx]
    return s_next, cost


class PathEnv:
    """The neural learner's only view of the world: sampled transitions, never the model."""

    def __init__(self, rng):
        self._rng = rng
        self._state = None  # no episode runs until reset() is called

    def reset(self):
        self._state = START
        return self._state

    def step(self, action_label):
        if self._state is None or self._state == TERMINAL:
            raise ValueError("no active episode: call reset() before step()")
        if action_label not in ACTIONS[self._state]:
            raise ValueError(f"action {action_label} is infeasible at state {self._state}")
        # Inside a method the bare name resolves to the module-level sampler, not to this method.
        next_state, cost = step(self._rng, self._state, action_label)
        self._state = next_state
        return next_state, cost, next_state == TERMINAL

    def feasible_actions(self, state):
        if state not in STATES:
            raise ValueError(f"unknown state {state}")
        # A copy, so a caller can never edit the table. ACTIONS has no entry for h, hence [].
        return list(ACTIONS.get(state, []))


# --------------------------------------------------------------------------
# 2. Model-access guard
# --------------------------------------------------------------------------
# The assignment forbids using the transition model, policy-iteration values or
# tabular optimal Q values as training targets. Every post-hoc function of section 3
# calls this guard first (the sampler step, above, is the only model reader that
# training may use), and every training function holds the flag
# in a try/finally, so an accidental call during training fails loudly. The flag
# is always back to False once a training function has returned or raised, so
# it carries nothing from one run to the next.
_training_active = False


def _forbid_during_training(name):
    if _training_active:
        raise RuntimeError(f"{name} uses the transition model and is forbidden during training")


# --------------------------------------------------------------------------
# 3. Policy iteration (exact, model-based, Assignment 2): post-hoc reference only
# --------------------------------------------------------------------------
def lookahead(V, s, a):
    _forbid_during_training("lookahead")
    return sum(p * (c + V[s_next]) for (s_next, p, c) in TRANSITIONS[s][a])


def policy_evaluation(policy, V_init=None, tol=1e-10):
    """Exact evaluation of a DETERMINISTIC policy (policy[s] is a single
    action string). Used to evaluate learned greedy policies against the
    known transition model, after training only. A single backward sweep is
    exact here because the induced transition graph is acyclic, so V_init and
    tol, kept from the Assignment 2 signature, are not used."""
    _forbid_during_training("policy_evaluation")
    V = {"h": 0.0}
    # d, e, f, g each have exactly one action, so their values are fixed
    # regardless of the policy at the branching states a, b, c.
    V["g"] = lookahead(V, "g", "Ah")
    V["f"] = lookahead(V, "f", "Ag")
    V["e"] = lookahead(V, "e", "Ag")
    V["d"] = lookahead(V, "d", "Ag")
    V["c"] = lookahead(V, "c", policy["c"])
    V["b"] = lookahead(V, "b", policy["b"])
    V["a"] = lookahead(V, "a", policy["a"])
    return V


def evaluate_general_policy(policy):
    """Exact evaluation of a possibly STOCHASTIC policy, used by policy
    iteration. policy[s] is either a single action string (deterministic)
    or a dict {action: probability} (stochastic), per
    V^pi(s) = sum_a pi(a|s) Q^pi(s,a), the weighted Bellman expectation
    equation, which reduces to the single-action lookahead when pi(.|s) is
    one-hot."""
    _forbid_during_training("evaluate_general_policy")

    def action_probs(entry):
        return entry if isinstance(entry, dict) else {entry: 1.0}

    V = {"h": 0.0}
    V["g"] = lookahead(V, "g", "Ah")
    V["f"] = lookahead(V, "f", "Ag")
    V["e"] = lookahead(V, "e", "Ag")
    V["d"] = lookahead(V, "d", "Ag")
    for s in ["c", "b", "a"]:
        V[s] = sum(p * lookahead(V, s, a) for a, p in action_probs(policy[s]).items())
    return V


def policy_improvement(V):
    _forbid_during_training("policy_improvement")
    new_policy = {}
    deltas = {}
    for s in ["a", "b", "c"]:
        qvals = {a: lookahead(V, s, a) for a in ACTIONS[s]}
        best_a = min(qvals, key=qvals.get)
        new_policy[s] = best_a
        deltas[s] = qvals
    return new_policy, deltas


def run_policy_iteration(log_path=None):
    """Policy iteration from the stochastic initial policy of Assignment 2. The
    sweep-by-sweep trace is written only if a log path is given."""
    _forbid_during_training("run_policy_iteration")
    lines = []
    policy = {
        "a": {"Ab": 0.60, "Ac": 0.40},
        "b": {"Ad": 0.70, "Ae": 0.30},
        "c": {"Ae": 0.50, "Af": 0.50},
    }
    lines.append("Initial stochastic policy pi0 (given in the assignment):")
    lines.append(f"  pi0 = {policy}\n")

    it = 0
    while True:
        it += 1
        V = evaluate_general_policy(policy)
        lines.append(f"--- Sweep {it}: Policy Evaluation ---")
        for s in STATES:
            lines.append(f"  V(%s) = %.4f" % (s, V[s]))
        new_policy, qvals = policy_improvement(V)
        lines.append(f"--- Sweep {it}: Policy Improvement (one-step lookahead) ---")
        for s in ["a", "b", "c"]:
            qs = ", ".join(f"Q({s},{a})={qvals[s][a]:.4f}" for a in ACTIONS[s])
            lines.append(f"  {s}: {qs}  -> greedy action = {new_policy[s]}")
        prev_actions = {s: (policy[s] if isinstance(policy[s], str) else None)
                        for s in ["a", "b", "c"]}
        changed = any(new_policy[s] != prev_actions[s] for s in ["a", "b", "c"])
        lines.append(f"  policy changed this sweep: {changed}\n")
        policy = new_policy
        if not changed:
            break

    V_star = policy_evaluation(policy)
    lines.append("=== Converged ===")
    lines.append(f"Optimal policy: {policy}")
    for s in STATES:
        lines.append(f"  V*(%s) = %.4f" % (s, V_star[s]))

    if log_path is not None:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return policy, V_star


def reference_values():
    """V*, Q* and pi* for reporting and post-hoc analysis. Never reaches a learner:
    training functions do not receive these numbers and the guard blocks this call."""
    _forbid_during_training("reference_values")
    pi_star, v_star = run_policy_iteration()
    q_star = {f"{s}|{a}": lookahead(v_star, s, a) for s, a in FEASIBLE_PAIRS}
    # d, e, f, g have one forced action each, which completes the policy.
    full_policy = {s: pi_star.get(s, ACTIONS[s][0]) for s in STATES if s != TERMINAL}
    return {"v_star": v_star, "q_star": q_star, "pi_star": full_policy}


def nominal_path(policy):
    """Path from a to h if every intended transition succeeds, and its cost, read from the
    edge cost of the intended branch. Post-hoc reporting only."""
    _forbid_during_training("nominal_path")
    s = START
    visited = [s]
    total_cost = 0
    while s != TERMINAL:
        action = policy[s]
        intended = action[1:]  # "Ab" intends b
        total_cost += next(c for (s_next, _, c) in TRANSITIONS[s][action] if s_next == intended)
        s = intended
        visited.append(s)
    return "->".join(visited), total_cost


def loss_noise_floor(replay_counts):
    """Expected MSE at Q = Q*: sum over (s,a) of freq(s,a) * Var[c + V*(s') | s,a]. Even with
    Q = Q* a sampled target still differs from Q*(s,a), because the realized branch varies, so
    the MSE cannot settle at zero. It is an expectation, not a hard bound: sampled branch rates
    that differ from 0.8 / 0.2 and a lagging target can put the loss somewhat below it, and it
    holds for the final buffer's mix of (s,a), so it describes the end of training. MSE only.
    Post-hoc analysis only: freq comes from the final replay buffer, the variance from the model."""
    _forbid_during_training("loss_noise_floor")
    v_star = reference_values()["v_star"]
    total = replay_counts.sum()
    floor = 0.0
    for s, a in FEASIBLE_PAIRS:
        freq = replay_counts[STATE_INDEX[s], ACTION_INDEX[a]] / total
        mean = sum(p * (c + v_star[s_next]) for (s_next, p, c) in TRANSITIONS[s][a])
        variance = sum(p * (c + v_star[s_next] - mean) ** 2 for (s_next, p, c) in TRANSITIONS[s][a])
        floor += freq * variance
    return float(floor)


# --------------------------------------------------------------------------
# 4. Neural Q function
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class NeuralConfig:
    """Every neural hyperparameter, in one place."""
    hidden: tuple = (32, 32)        # widths of the ReLU hidden layers
    capacity: int = 10000           # replay capacity, in transitions
    warmup: int = 200               # transitions stored before the first learning step
    batch_size: int = 32            # mini-batch size
    lr: float = 1e-3                # Adam learning rate
    target_update: int = 100        # M: learning steps between hard copies to the target
    gamma: float = 1.0              # discount factor
    loss: str = "mse"               # "mse", or "huber" (used only by the sweep)
    huber_delta: float = 1.0        # threshold of the Huber loss
    episodes: int = 5000            # training episodes per seed
    eps_start: float = 1.0          # epsilon at episode 1
    eps_end: float = 0.05           # epsilon floor
    eps_decay_episodes: int = 4000  # episode at which epsilon reaches the floor


class QNetwork(nn.Module):
    """Input: one-hot state over a..h. Output: 7 linear values in the order of ACTION_LABELS,
    each estimating the expected total cost-to-go of that action, so smaller is better. Linear
    outputs because cost-to-go is an unbounded positive quantity that no squashing should limit."""

    def __init__(self, hidden=(32, 32)):
        super().__init__()
        layers = []
        in_size = N_STATES
        for size in hidden:
            layers += [nn.Linear(in_size, size), nn.ReLU()]
            in_size = size
        layers.append(nn.Linear(in_size, N_ACTIONS))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def weights_sha256(net):
    """Fingerprint of the exact weights, used to prove that runs are reproducible. It repeats
    bit for bit only on the same OS, torch build and CPU instruction set (all recorded in meta);
    elsewhere the weights differ in the last digits while the reported statistics agree."""
    digest = hashlib.sha256()
    for name, tensor in net.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def encode(state_index):
    """One-hot input for a state index (int) or a batch of indices (1-D int tensor)."""
    return F.one_hot(torch.as_tensor(state_index), N_STATES).float()


def feasibility_mask(env):
    """Boolean N_STATES x N_ACTIONS mask, built only from the environment's public API.
    The row of h is all False because no action exists there."""
    mask = torch.zeros(N_STATES, N_ACTIONS, dtype=torch.bool)
    for state in STATES:
        for label in env.feasible_actions(state):
            mask[STATE_INDEX[state], ACTION_INDEX[label]] = True
    return mask


def all_state_q_values(net):
    """Q values of every state as an N_STATES x N_ACTIONS array. Consumes no randomness."""
    with torch.no_grad():
        return net(encode(torch.arange(N_STATES))).numpy()


# --------------------------------------------------------------------------
# 5. Action selection
# --------------------------------------------------------------------------
def masked_argmin(q_values, feasible, rng):
    """Index of the smallest feasible value; exact ties are broken uniformly at random.
    An infeasible value is never considered, however attractive it is. rng is only
    drawn from when a tie really exists."""
    q_values = np.asarray(q_values)
    candidates = np.flatnonzero(np.asarray(feasible))
    if candidates.size == 0:
        raise ValueError("no feasible action to select")
    feasible_values = q_values[candidates]
    best = candidates[feasible_values == feasible_values.min()]
    # Empty only if a diverged network produced nan; then no value is better than another.
    # Defensive: no sweep setting has reached it, but a diverged run must finish, not crash.
    if best.size == 0:
        best = candidates
    if best.size == 1:
        return int(best[0])
    return int(best[rng.integers(best.size)])


def epsilon_at(episode, eps_start, eps_end, decay_episodes):
    """Epsilon of a 1-indexed episode: linear from eps_start at episode 1 to eps_end at
    episode decay_episodes, then constant. Epsilon changes per episode, not per step."""
    frac = 1.0 if decay_episodes <= 1 else min(1.0, (episode - 1) / (decay_episodes - 1))
    return eps_start + frac * (eps_end - eps_start)


def select_action(online_net, env, state, mask, eps, rng):
    """Epsilon-greedy over the feasible actions."""
    labels = env.feasible_actions(state)
    if len(labels) == 1:
        return labels[0]  # no decision to make, so no draw, as in Assignment 2
    if rng.random() < eps:
        return labels[rng.integers(len(labels))]
    index = STATE_INDEX[state]
    with torch.no_grad():
        q_values = online_net(encode(index))
    return ACTION_LABELS[masked_argmin(q_values, mask[index], rng)]


# --------------------------------------------------------------------------
# 6. Replay buffer
# --------------------------------------------------------------------------
class ReplayBuffer:
    """Preallocated ring buffer of (state, action, cost, next state, done) index tuples."""

    def __init__(self, capacity, rng):
        self.capacity = capacity
        self._rng = rng
        self.states = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.costs = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros(capacity, dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=bool)
        self._next_slot = 0
        self._size = 0

    def __len__(self):
        return self._size

    def add(self, state, action, cost, next_state, done):
        i = self._next_slot
        self.states[i] = state
        self.actions[i] = action
        self.costs[i] = cost
        self.next_states[i] = next_state
        self.dones[i] = done
        # When full, the oldest transition is the next one overwritten.
        self._next_slot = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        # Uniform with replacement, over the filled part only: the zeros in unused slots
        # are not transitions and must never be drawn.
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (torch.from_numpy(self.states[idx]),
                torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.costs[idx]),
                torch.from_numpy(self.next_states[idx]),
                torch.from_numpy(self.dones[idx]))

    def pair_counts(self):
        """Stored transitions per (state, action), as an N_STATES x N_ACTIONS array."""
        counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
        np.add.at(counts, (self.states[:self._size], self.actions[:self._size]), 1)
        return counts


# --------------------------------------------------------------------------
# 7. Neural Q-learning
# --------------------------------------------------------------------------
def compute_targets(target_net, costs, next_states, dones, mask, gamma):
    """y = c if done, else c + gamma * min over feasible a' of Q_target(s', a').
    Infeasible outputs are filled with +inf so the min can never pick them. The row of h is
    entirely infeasible, so its min is +inf; that must be replaced by the terminal value 0 with
    torch.where, because zeroing by multiplication would give inf * 0 = nan."""
    with torch.no_grad():
        q_next = target_net(encode(next_states)).masked_fill(~mask[next_states], float("inf"))
        best_next = q_next.min(dim=1).values
        bootstrap = torch.where(dones, torch.zeros_like(best_next), best_next)
        return costs + gamma * bootstrap


def learning_step(online_net, target_net, optimizer, buffer, mask, cfg):
    """One mini-batch update of the online network. Returns the batch loss."""
    states, actions, costs, next_states, dones = buffer.sample(cfg.batch_size)
    targets = compute_targets(target_net, costs, next_states, dones, mask, cfg.gamma)
    # gather keeps only Q(s_i, a_i). Stored actions are always feasible, so an infeasible
    # output is never part of the loss and is never trained.
    q_taken = online_net(encode(states)).gather(1, actions.unsqueeze(1)).squeeze(1)
    if q_taken.shape != targets.shape:  # a shape mismatch would silently broadcast in the loss
        raise ValueError(f"shape mismatch: Q {q_taken.shape} vs y {targets.shape}")
    if cfg.loss == "mse":
        loss = F.mse_loss(q_taken, targets)
    elif cfg.loss == "huber":
        # Mean of 0.5 * d^2 if |d| <= delta, else delta * (|d| - 0.5 * delta), where d = y - Q.
        loss = F.huber_loss(q_taken, targets, delta=cfg.huber_delta)
    else:
        raise ValueError(f"unknown loss {cfg.loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def expected_counts(cfg):
    """Transitions, learning steps and target syncs a run must produce. Every episode has
    exactly 4 transitions and learning starts at the transition that reaches warmup."""
    n_transitions = 4 * cfg.episodes
    n_learning_steps = max(0, n_transitions - cfg.warmup + 1)
    return n_transitions, n_learning_steps, n_learning_steps // cfg.target_update


def train_neural(seed, cfg):
    """One completely independent training run: every network, optimizer, buffer and generator
    is built here, so nothing carries over from an earlier run in the same process."""
    # A buffer smaller than the warm-up could never start learning, and a warm-up of 0 would act
    # like 1 and break expected_counts: refuse both instead of training silently.
    if not 1 <= cfg.warmup <= cfg.capacity:
        raise ValueError(f"need 1 <= warmup <= capacity, got warmup {cfg.warmup} and "
                         f"capacity {cfg.capacity}")
    global _training_active
    _training_active = True
    try:
        # The global generators are seeded even though the learner has its own, because torch
        # draws its weight initialization from the global torch generator.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)  # one thread keeps the floating-point results reproducible
        env_stream, explore_stream, replay_stream = np.random.SeedSequence(seed).spawn(3)
        env = PathEnv(np.random.default_rng(env_stream))
        explore_rng = np.random.default_rng(explore_stream)
        buffer = ReplayBuffer(cfg.capacity, np.random.default_rng(replay_stream))

        mask = feasibility_mask(env)
        mask_np = mask.numpy()
        online_net = QNetwork(cfg.hidden)
        target_net = copy.deepcopy(online_net)
        for param in target_net.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.Adam(online_net.parameters(), lr=cfg.lr)

        episode_costs = np.zeros(cfg.episodes)
        epsilons = np.zeros(cfg.episodes)
        q_hist = np.zeros((cfg.episodes, int(mask.sum())), dtype=np.float32)
        losses = []
        n_transitions = 0
        n_learning_steps = 0
        n_target_syncs = 0

        # The clock starts here because the first Adam built in a process spends about a second
        # on lazy imports, which would otherwise be charged to whichever seed happens to run first.
        start_time = time.perf_counter()
        for episode in range(1, cfg.episodes + 1):
            eps = epsilon_at(episode, cfg.eps_start, cfg.eps_end, cfg.eps_decay_episodes)
            state = env.reset()
            total_cost = 0.0
            done = False
            while not done:
                action = select_action(online_net, env, state, mask, eps, explore_rng)
                next_state, cost, done = env.step(action)
                buffer.add(STATE_INDEX[state], ACTION_INDEX[action], cost,
                           STATE_INDEX[next_state], done)
                total_cost += cost
                n_transitions += 1

                if len(buffer) >= cfg.warmup:
                    losses.append(learning_step(online_net, target_net, optimizer, buffer,
                                                mask, cfg))
                    n_learning_steps += 1
                    # M counts learning steps, not environment steps or episodes.
                    if n_learning_steps % cfg.target_update == 0:
                        target_net.load_state_dict(online_net.state_dict())
                        n_target_syncs += 1

                state = next_state

            episode_costs[episode - 1] = total_cost
            epsilons[episode - 1] = eps
            # Boolean indexing is row-major, which is the documented order of the 10 columns.
            q_hist[episode - 1] = all_state_q_values(online_net)[mask_np]

        return {
            "online_net": online_net,
            "mask": mask,
            "episode_costs": episode_costs,
            "epsilons": epsilons,
            "losses": np.array(losses, dtype=np.float32),
            "q_hist": q_hist,
            "replay_counts": buffer.pair_counts(),
            "n_transitions": n_transitions,
            "n_learning_steps": n_learning_steps,
            "n_target_syncs": n_target_syncs,
            "weights_sha256": weights_sha256(online_net),
            "wall_clock_s": time.perf_counter() - start_time,
        }
    finally:
        _training_active = False


# --------------------------------------------------------------------------
# 8. Tabular Q-learning baseline (Assignment 2)
# --------------------------------------------------------------------------
def epsilon_schedule(episode, eps_start=1.0, eps_min=0.05, decay_episodes=2000):
    frac = min(1.0, episode / decay_episodes)
    return eps_start + frac * (eps_min - eps_start)


def run_q_learning(seed, n_episodes=5000, alpha=0.1, eps_start=1.0, eps_min=0.05,
                   decay_episodes=2000):
    """The Assignment 2 learner: same draws in the same order from one generator, so its Q
    tables and episode costs equal Assignment 2's exactly. Added only: the guard flag, the
    wall clock, and logging that consumes no randomness."""
    global _training_active
    _training_active = True
    start_time = time.perf_counter()
    try:
        rng = np.random.default_rng(seed)
        Q = {s: {a: 0.0 for a in ACTIONS[s]} for s in STATES if s != TERMINAL}

        episode_costs = np.zeros(n_episodes)
        epsilons = np.zeros(n_episodes)
        q_hist = np.zeros((n_episodes, len(FEASIBLE_PAIRS)))
        n_transitions = 0

        for ep in range(n_episodes):
            eps = epsilon_schedule(ep, eps_start, eps_min, decay_episodes)
            s = "a"
            total_cost = 0.0
            while s != TERMINAL:
                acts = ACTIONS[s]
                if len(acts) == 1:
                    a = acts[0]
                else:
                    if rng.random() < eps:
                        a = acts[rng.integers(len(acts))]
                    else:
                        # greedy w.r.t. current Q, ties broken uniformly at random
                        qvals = np.array([Q[s][act] for act in acts])
                        best = np.flatnonzero(qvals == qvals.min())
                        a = acts[rng.choice(best)]

                s_next, cost = step(rng, s, a)
                total_cost += cost
                n_transitions += 1

                if s_next == TERMINAL:
                    bootstrap = 0.0
                else:
                    bootstrap = min(Q[s_next].values())

                td_target = cost + bootstrap
                td_error = td_target - Q[s][a]
                Q[s][a] += alpha * td_error

                s = s_next

            episode_costs[ep] = total_cost
            epsilons[ep] = eps
            q_hist[ep] = [Q[st][act] for st, act in FEASIBLE_PAIRS]

        return {
            "Q": Q,
            "episode_costs": episode_costs,
            "epsilons": epsilons,
            "q_hist": q_hist,
            "n_transitions": n_transitions,
            "wall_clock_s": time.perf_counter() - start_time,
        }
    finally:
        _training_active = False


def tabular_settings(cfg):
    """The two baselines. tabular_a2 is the run submitted in Assignment 2 (5000 episodes, decay
    over the first 2000), so it keeps those settings even if the neural defaults change; only a
    shortened test run (--episodes) scales it, with the decay over 40% of the episodes.
    tabular_matched has the neural agent's length and epsilon in every episode: the Assignment 2
    schedule is zero-indexed, hence one less than eps_decay_episodes."""
    if cfg.episodes == NeuralConfig().episodes:
        a2_episodes, a2_decay = 5000, 2000
    else:
        a2_episodes, a2_decay = cfg.episodes, max(1, round(0.4 * cfg.episodes))
    return {
        "tabular_a2": {
            "n_episodes": a2_episodes,
            "alpha": 0.1,
            "eps_start": 1.0,
            "eps_min": 0.05,
            "decay_episodes": a2_decay,
        },
        "tabular_matched": {
            "n_episodes": cfg.episodes,
            "alpha": 0.1,
            "eps_start": cfg.eps_start,
            "eps_min": cfg.eps_end,
            "decay_episodes": max(1, cfg.eps_decay_episodes - 1),
        },
    }


def table_to_matrix(Q):
    """A tabular Q as an N_STATES x N_ACTIONS array, comparable with a network's outputs."""
    matrix = np.full((N_STATES, N_ACTIONS), np.inf)
    for s, a in FEASIBLE_PAIRS:
        matrix[STATE_INDEX[s], ACTION_INDEX[a]] = Q[s][a]
    return matrix


# --------------------------------------------------------------------------
# 9. Evaluation and post-hoc analysis
# --------------------------------------------------------------------------
EVAL_SEED_ROOT = 598003
EVAL_EPISODES = 1000
EVAL_MAX_STEPS = 50  # guard: nontermination would show up as < 100% reaching h, not as a hang
Z_95 = 1.96          # normal quantile of a two-sided 95% confidence interval


def evaluation_stream(seed):
    """Environment noise of the evaluation of a seed, separate from every training stream.
    Each method starts a fresh copy, so all methods of a seed face the same noise (common
    random numbers) and their results can be compared pairwise."""
    return np.random.default_rng(np.random.SeedSequence([EVAL_SEED_ROOT, seed]))


def evaluation_tie_stream(seed):
    """Tie-breaking of evaluated policies. Kept apart from the noise above so that an exact tie
    in one method cannot shift the noise and break the pairing."""
    return np.random.default_rng(np.random.SeedSequence([EVAL_SEED_ROOT, seed, 1]))


def greedy_policy(q_matrix, mask, tie_rng):
    """Greedy action at all 7 nonterminal states."""
    return {s: ACTION_LABELS[masked_argmin(q_matrix[i], mask[i], tie_rng)]
            for i, s in enumerate(STATES) if s != TERMINAL}


def neural_actor(net, mask, tie_rng):
    """The policy of a trained network: one forward pass and one masked argmin per decision."""
    def act(state):
        index = STATE_INDEX[state]
        with torch.no_grad():
            q_values = net(encode(index))
        return ACTION_LABELS[masked_argmin(q_values, mask[index], tie_rng)]
    return act


def table_actor(policy):
    """The policy of a tabular learner, as a lookup of its greedy actions."""
    def act(state):
        return policy[state]
    return act


def evaluate_policy(act, seed, n_episodes=EVAL_EPISODES, max_steps=EVAL_MAX_STEPS):
    """Stochastic episodes with exploration off, on the evaluation stream of this seed."""
    env = PathEnv(evaluation_stream(seed))
    totals = np.zeros(n_episodes)
    n_reached = 0
    for i in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            state, cost, done = env.step(act(state))
            totals[i] += cost
            steps += 1
        n_reached += int(done)
    values, counts = np.unique(totals, return_counts=True)
    std = float(np.std(totals, ddof=1))
    return {
        "n_episodes": n_episodes,
        "mean": float(np.mean(totals)),
        "std": std,
        "min": float(np.min(totals)),
        "max": float(np.max(totals)),
        "pct_reach_h": 100.0 * n_reached / n_episodes,
        "ci95_half_width": float(Z_95 * std / np.sqrt(n_episodes)),
        "cost_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
    }


def rolling_mean(values, window):
    """Trailing rolling mean. Returns (k, mean) where k is the 1-indexed position of the last
    point of each window, so the value at k averages the points k-window+1 .. k."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.array([], dtype=int), np.array([])
    means = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.arange(window, len(values) + 1), means


def stability_metrics(q_hist, reference):
    """Post-hoc stability metrics from the logged Q values, using no randomness. The slices
    [-1000:] take every episode of a run of at most 1000 episodes; that window then includes
    the early learning phase, so for such runs compare first_stable_episode rather than
    optimal_frac_last_1000 and switches_last_1000 with those of longer runs."""
    q_star = np.array([reference["q_star"][f"{s}|{a}"] for s, a in FEASIBLE_PAIRS])
    greedy_ok = np.ones(len(q_hist), dtype=bool)
    choices = []
    for s in BRANCHING_STATES:
        columns = [FEASIBLE_PAIRS.index((s, a)) for a in ACTIONS[s]]
        best = FEASIBLE_PAIRS.index((s, reference["pi_star"][s]))
        other = columns[0] if columns[1] == best else columns[1]
        # Strict, so an exact tie does not count as having learned the optimal action.
        greedy_ok &= q_hist[:, best] < q_hist[:, other]
        choices.append(np.argmin(q_hist[:, columns], axis=1))
    triple = np.stack(choices, axis=1)[-1000:]
    switches = int(np.sum(np.any(triple[1:] != triple[:-1], axis=1)))

    # The first episode from which greedy_ok holds through to the end of the run.
    not_ok = np.flatnonzero(~greedy_ok)
    if not greedy_ok[-1]:
        first_stable = None
    elif not_ok.size == 0:
        first_stable = 1
    else:
        first_stable = int(not_ok[-1]) + 2  # 0-indexed failure -> 1-indexed episode after it

    q_rmse_curve = np.sqrt(np.mean((q_hist - q_star) ** 2, axis=1))
    return {
        "greedy_ok": greedy_ok,
        "optimal_frac_last_1000": float(np.mean(greedy_ok[-1000:])),
        "switches_last_1000": switches,
        "first_stable_episode": first_stable,
        "q_rmse": float(q_rmse_curve[-1]),
        "q_rmse_curve": q_rmse_curve,
    }


def method_record(seed, policy, act, run, n_learning_steps, reference):
    """The per-seed record shared by all methods."""
    path, nominal_cost = nominal_path(policy)
    exact_cost = policy_evaluation(policy)["a"]
    gap = exact_cost - reference["v_star"]["a"]
    metrics = stability_metrics(run["q_hist"], reference)
    return {
        "policy": policy,
        "nominal_path": path,
        "nominal_cost": nominal_cost,
        "eval": evaluate_policy(act, seed),
        "exact_expected_cost": exact_cost,
        "gap": gap,
        "is_optimal": bool(abs(gap) < 1e-6),
        "q_final": {f"{s}|{a}": float(run["q_hist"][-1, i])
                    for i, (s, a) in enumerate(FEASIBLE_PAIRS)},
        "q_rmse": metrics["q_rmse"],
        "optimal_frac_last_1000": metrics["optimal_frac_last_1000"],
        "switches_last_1000": metrics["switches_last_1000"],
        "first_stable_episode": metrics["first_stable_episode"],
        "rolling_cost_last_100": float(np.mean(run["episode_costs"][-100:])),
        "n_transitions": run["n_transitions"],
        "n_learning_steps": n_learning_steps,
        "wall_clock_s": run["wall_clock_s"],
    }


# --------------------------------------------------------------------------
# 10. Self-checks
# --------------------------------------------------------------------------
# Each check_* function reports through record(name, passed, detail). They exercise the same
# functions the experiments use: where a check must observe the real training loop, it swaps
# one module-level function for a small spy for the duration of the run.

def _with_replaced_global(name, replacement, body):
    """Run body() while the module-level function called name is replaced."""
    original = globals()[name]
    globals()[name] = replacement
    try:
        return body()
    finally:
        globals()[name] = original


def _assignment_config():
    """The starting configuration named in the assignment. The checks use it instead of the
    defaults, which may change after the sweep without making any check wrong."""
    return NeuralConfig(hidden=(32, 32), capacity=10000, warmup=200, batch_size=32, lr=1e-3,
                        target_update=100, gamma=1.0, loss="mse", huber_delta=1.0, episodes=5000,
                        eps_start=1.0, eps_end=0.05, eps_decay_episodes=4000)


def _short_config(**changes):
    """A run short enough for the self-checks."""
    settings = dict(episodes=60, warmup=50, target_update=10, eps_decay_episodes=48)
    settings.update(changes)
    return dataclasses.replace(_assignment_config(), **settings)


def _test_mask():
    return feasibility_mask(PathEnv(np.random.default_rng(0)))


def _stub_network(table):
    """A callable with a network's interface that returns a row of a fixed table per state."""
    def net(x):
        return table[x.argmax(dim=-1)]
    return net


def check_transition_table(record):
    record("STATES are a..h with terminal h and start a",
           STATES == list("abcdefgh") and TERMINAL == "h" and PathEnv(None).reset() == "a")
    record("TRANSITIONS has one entry per nonterminal state and matches ACTIONS",
           sorted(TRANSITIONS) == list("abcdefg")
           and all(sorted(TRANSITIONS[s]) == sorted(ACTIONS[s]) for s in TRANSITIONS))
    outcomes = [o for by_action in TRANSITIONS.values() for o in by_action.values()]
    record("branch probabilities sum to 1 for every (s,a)",
           all(abs(sum(p for _, p, _ in o) - 1.0) < 1e-12 for o in outcomes))
    record("branching actions are 0.8 intended / 0.2 alternative, forced actions 1.0",
           all(([p for _, p, _ in TRANSITIONS[s][a]] == [0.8, 0.2]
                if s in BRANCHING_STATES else [p for _, p, _ in TRANSITIONS[s][a]] == [1.0])
               and TRANSITIONS[s][a][0][0] == a[1:] for s, a in FEASIBLE_PAIRS))
    expected_edges = {("a", "b"): 2, ("a", "c"): 2, ("b", "d"): 3, ("b", "e"): 2, ("c", "e"): 5,
                      ("c", "f"): 4, ("d", "g"): 3, ("e", "g"): 3, ("f", "g"): 2, ("g", "h"): 1}
    edges_ok = all(expected_edges.get((s, s_next)) == c
                   for s, by_action in TRANSITIONS.items()
                   for o in by_action.values() for s_next, _, c in o)
    record("every branch carries the edge cost of the graph of Assignment 2", edges_ok)


def check_feasible_sets(record):
    expected = {"a": ["Ab", "Ac"], "b": ["Ad", "Ae"], "c": ["Ae", "Af"], "d": ["Ag"],
                "e": ["Ag"], "f": ["Ag"], "g": ["Ah"], "h": []}
    env = PathEnv(np.random.default_rng(0))
    record("feasible_actions gives the expected labels, empty at h",
           all(env.feasible_actions(s) == labels for s, labels in expected.items()))
    record("ACTION_LABELS is Ab, Ac, Ad, Ae, Af, Ag, Ah and covers every action",
           ACTION_LABELS == ["Ab", "Ac", "Ad", "Ae", "Af", "Ag", "Ah"]
           and sorted({a for acts in ACTIONS.values() for a in acts}) == ACTION_LABELS)
    record("FEASIBLE_PAIRS is the documented order of the 10 pairs",
           [f"{s}|{a}" for s, a in FEASIBLE_PAIRS]
           == ["a|Ab", "a|Ac", "b|Ad", "b|Ae", "c|Ae", "c|Af", "d|Ag", "e|Ag", "f|Ag", "g|Ah"])
    returned = env.feasible_actions("a")
    returned.append("Zz")  # editing a returned list must not edit the environment
    record("PathEnv's public API is exactly reset, step and feasible_actions, and hands out copies",
           [name for name in dir(PathEnv) if not name.startswith("_")]
           == ["feasible_actions", "reset", "step"] and env.feasible_actions("a") == ["Ab", "Ac"])


def check_empirical_branches(record):
    rng = np.random.default_rng(12345)
    n_draws = 4000
    worst_error = 0.0
    costs_ok = True
    for s, a in FEASIBLE_PAIRS:
        branches = {s_next: (p, c) for s_next, p, c in TRANSITIONS[s][a]}
        counts = {s_next: 0 for s_next in branches}
        for _ in range(n_draws):
            s_next, cost = step(rng, s, a)
            counts[s_next] += 1
            costs_ok = costs_ok and cost == branches[s_next][1]
        for s_next, (p, _) in branches.items():
            worst_error = max(worst_error, abs(counts[s_next] / n_draws - p))
    record("empirical frequency of every (s,a) branch is within 0.03 of its probability",
           worst_error < 0.03, f"worst error {worst_error:.4f}")
    record("the realized cost matches the realized branch", costs_ok)

    # The same frequencies through PathEnv.step, the learner's only door to the world, over
    # episodes with random actions (at least about 2000 draws per branching pair).
    env = PathEnv(np.random.default_rng(54321))
    seen = {pair: {} for pair in FEASIBLE_PAIRS}
    for _ in range(8000):
        state = env.reset()
        done = False
        while not done:
            labels = env.feasible_actions(state)
            label = labels[rng.integers(len(labels))]
            next_state, _, done = env.step(label)
            seen[(state, label)][next_state] = seen[(state, label)].get(next_state, 0) + 1
            state = next_state
    worst_error = max(abs(seen[(s, a)].get(s_next, 0) / sum(seen[(s, a)].values()) - p)
                      for s, a in FEASIBLE_PAIRS for s_next, p, _ in TRANSITIONS[s][a])
    record("through PathEnv.step as well, every branch frequency is within 0.03 of its probability",
           worst_error < 0.03, f"worst error {worst_error:.4f}")


def check_episodes(record):
    rng = np.random.default_rng(1)
    env = PathEnv(rng)
    lengths_ok = True
    done_ok = True
    totals = set()
    for _ in range(2000):
        state = env.reset()
        total = 0
        n_steps = 0
        done = False
        while not done:
            labels = env.feasible_actions(state)
            state, cost, done = env.step(labels[rng.integers(len(labels))])
            total += cost
            n_steps += 1
            done_ok = done_ok and done == (state == "h")
        lengths_ok = lengths_ok and n_steps == 4
        totals.add(total)
    record("every episode reaches h in exactly 4 steps", lengths_ok and done_ok)
    record("every episode has a total cost in {8, 9, 11}", totals == {8, 9, 11},
           str(sorted(totals)))


def check_env_errors(record):
    def raises_value_error(call):
        try:
            call()
        except ValueError:
            return True
        return False

    env = PathEnv(np.random.default_rng(2))
    record("step raises before the first reset", raises_value_error(lambda: env.step("Ab")))
    env.reset()
    record("step raises on an infeasible action", raises_value_error(lambda: env.step("Ah")))
    record("step raises on an unknown action", raises_value_error(lambda: env.step("Zz")))
    state, _, _ = env.step("Ab")
    while state != "h":
        state, _, _ = env.step(env.feasible_actions(state)[0])
    record("step raises after the episode has ended, whatever the action",
           all(raises_value_error(lambda label=label: env.step(label)) for label in ACTION_LABELS))
    record("reset starts a new episode at a", env.reset() == "a" and env.step("Ab")[2] is False)


def check_mask_and_argmin(record):
    mask = _test_mask()
    expected = torch.zeros(N_STATES, N_ACTIONS, dtype=torch.bool)
    for s, a in FEASIBLE_PAIRS:
        expected[STATE_INDEX[s], ACTION_INDEX[a]] = True
    record("mask is 8 x 7, matches the feasible sets, and the h row is all False",
           mask.shape == (8, 7) and torch.equal(mask, expected) and not mask[7].any()
           and int(mask.sum()) == 10)
    rng = np.random.default_rng(3)
    all_ok = True
    for _ in range(3000):
        index = int(rng.integers(N_STATES - 1))
        q_values = torch.tensor(rng.uniform(0.0, 10.0, size=N_ACTIONS), dtype=torch.float32)
        q_values[~mask[index]] = -1e9  # the most attractive value is always an infeasible one
        chosen = masked_argmin(q_values, mask[index], rng)
        all_ok = (all_ok and bool(mask[index, chosen])
                  and q_values[chosen] == q_values[mask[index]].min())
    record("masked_argmin picks the best feasible action, never a more attractive infeasible one",
           all_ok)
    try:
        masked_argmin(torch.zeros(N_ACTIONS), mask[STATE_INDEX["h"]], rng)
        record("masked_argmin refuses a state with no feasible action", False)
    except ValueError:
        record("masked_argmin refuses a state with no feasible action", True)
    nan_row = torch.full((N_ACTIONS,), float("nan"))
    record("masked_argmin stays on a feasible action if a diverged network outputs nan",
           bool(mask[0, masked_argmin(nan_row, mask[0], rng)]))
    one_hot = encode(2)
    batch = encode(torch.tensor([0, 7]))
    record("encode gives one-hot float vectors over 8 states",
           one_hot.tolist() == [0, 0, 1, 0, 0, 0, 0, 0] and batch.shape == (2, 8)
           and batch[1, 7] == 1 and batch.dtype == torch.float32)


def check_action_selection(record):
    mask = _test_mask()
    rng = np.random.default_rng(4)
    a_index = STATE_INDEX["a"]
    tied = torch.full((N_ACTIONS,), -1e9)  # infeasible outputs are the lowest
    tied[ACTION_INDEX["Ab"]] = 3.0
    tied[ACTION_INDEX["Ac"]] = 3.0
    picks = [masked_argmin(tied, mask[a_index], rng) for _ in range(4000)]
    frac = picks.count(ACTION_INDEX["Ab"]) / len(picks)
    record("exact ties split about 50/50", abs(frac - 0.5) < 0.03 and set(picks) == {0, 1},
           f"first action {frac:.3f}")
    global_before = np.random.get_state()
    tie_rng = np.random.default_rng(8)
    tie_rng_before = tie_rng.bit_generator.state
    masked_argmin(tied, mask[a_index], tie_rng)
    global_after = np.random.get_state()
    record("an exact tie draws from the dedicated generator, never from the global one",
           tie_rng.bit_generator.state != tie_rng_before
           and np.array_equal(global_after[1], global_before[1])
           and global_after[2] == global_before[2])

    table = torch.zeros(N_STATES, N_ACTIONS)
    table[a_index, ACTION_INDEX["Ab"]] = 1.0
    table[a_index, ACTION_INDEX["Ac"]] = 2.0
    stub = _stub_network(table)
    env = PathEnv(np.random.default_rng(0))
    before = rng.bit_generator.state
    chosen = masked_argmin(table[a_index], mask[a_index], rng)
    record("masked_argmin consumes no draw without a tie",
           chosen == ACTION_INDEX["Ab"] and rng.bit_generator.state == before)
    chosen = select_action(stub, env, "d", mask, 1.0, rng)
    record("a state with one feasible action consumes no random draw",
           chosen == "Ag" and rng.bit_generator.state == before)
    record("epsilon 0 is greedy",
           {select_action(stub, env, "a", mask, 0.0, rng) for _ in range(300)} == {"Ab"})
    picks = [select_action(stub, env, "a", mask, 1.0, rng) for _ in range(4000)]
    frac = picks.count("Ab") / len(picks)
    record("epsilon 1 is uniform over the feasible actions",
           abs(frac - 0.5) < 0.03 and set(picks) == {"Ab", "Ac"}, f"first action {frac:.3f}")
    picks = [select_action(stub, env, "a", mask, 0.5, rng) for _ in range(4000)]
    frac = picks.count("Ab") / len(picks)
    record("epsilon 0.5 picks the greedy action about 75% of the time", abs(frac - 0.75) < 0.03,
           f"greedy action {frac:.3f}")


def check_compute_targets(record):
    mask = _test_mask()
    index_b, index_c, index_h = STATE_INDEX["b"], STATE_INDEX["c"], STATE_INDEX["h"]
    table = torch.full((N_STATES, N_ACTIONS), 100.0)
    table[index_b, ACTION_INDEX["Ad"]] = 5.0
    table[index_b, ACTION_INDEX["Ae"]] = 3.0
    table[index_b, ACTION_INDEX["Ah"]] = -1e6  # infeasible at b and hugely negative
    table[index_c, ACTION_INDEX["Ae"]] = 7.0
    table[index_c, ACTION_INDEX["Af"]] = 6.0
    table[index_c, ACTION_INDEX["Ab"]] = -1e6  # infeasible at c and hugely negative
    table[index_h, :] = -1e6                   # nothing is feasible at h
    target_stub = _stub_network(table)

    costs = torch.tensor([2.0, 4.0, 1.0, 1.0])
    next_states = torch.tensor([index_b, index_c, index_h, index_h])
    dones = torch.tensor([False, False, True, True])
    y = compute_targets(target_stub, costs, next_states, dones, mask, 1.0)
    record("terminal transitions give y == c exactly, with no nan",
           torch.equal(y[2:], costs[2:]) and not torch.isnan(y).any())
    record("nonterminal y uses the min over feasible actions, ignoring a hugely negative "
           "infeasible output", y[:2].tolist() == [5.0, 10.0])
    record("y is 1-D, has the shape of c and carries no gradient",
           y.shape == costs.shape and not y.requires_grad)
    y = compute_targets(target_stub, costs, next_states, dones, mask, 0.5)
    record("gamma scales the bootstrap only", y.tolist() == [3.5, 7.0, 1.0, 1.0])
    y = compute_targets(target_stub, costs, next_states, dones, mask, 0.0)
    record("gamma 0 gives y == c with no inf * 0 = nan", torch.equal(y, costs))
    unfrozen = QNetwork()  # trainable parameters, so only torch.no_grad can keep y out of the graph
    record("targets are computed under torch.no_grad",
           not compute_targets(unfrozen, costs, next_states, dones, mask, 1.0).requires_grad)


def check_learning_step(record):
    mask = _test_mask()
    index_a, index_b = STATE_INDEX["a"], STATE_INDEX["b"]
    action_ab = ACTION_INDEX["Ab"]
    torch.manual_seed(7)
    online_net = QNetwork()
    target_net = QNetwork()
    for param in target_net.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)
    base = dataclasses.replace(_assignment_config(), batch_size=4)

    # One stored transition: the returned loss exposes which network bootstrapped.
    buffer = ReplayBuffer(1, np.random.default_rng(0))
    buffer.add(index_a, action_ab, 2.0, index_b, False)
    feasible_at_b = [ACTION_INDEX["Ad"], ACTION_INDEX["Ae"]]
    with torch.no_grad():
        q_taken = online_net(encode(index_a))[action_ab]
        from_target = (2.0 + target_net(encode(index_b))[feasible_at_b].min() - q_taken) ** 2
        from_online = (2.0 + online_net(encode(index_b))[feasible_at_b].min() - q_taken) ** 2
    loss = learning_step(online_net, target_net, optimizer, buffer, mask, base)
    record("learning_step bootstraps from the target network, not the online network",
           abs(loss - from_target.item()) < 1e-4 and abs(loss - from_online.item()) > 1e-3,
           f"loss {loss:.5f}, target {from_target.item():.5f}, online {from_online.item():.5f}")
    output_grad = online_net.layers[-1].weight.grad
    other_actions = [j for j in range(N_ACTIONS) if j != action_ab]
    record("only the output of the taken action receives gradient",
           output_grad[other_actions].abs().sum() == 0 and output_grad[action_ab].abs().sum() > 0)

    # The Huber option on a residual beyond delta: delta * (|d| - 0.5 * delta).
    target_net.load_state_dict(online_net.state_dict())
    with torch.no_grad():
        q_taken = online_net(encode(index_a))[action_ab]
        residual = abs(2.0 + target_net(encode(index_b))[feasible_at_b].min() - q_taken).item()
    loss = learning_step(online_net, target_net, optimizer, buffer, mask,
                         dataclasses.replace(base, loss="huber", huber_delta=0.5))
    record("the huber loss follows its definition", abs(loss - 0.5 * (residual - 0.25)) < 1e-5,
           f"loss {loss:.5f}, residual {residual:.5f}")
    try:
        learning_step(online_net, target_net, optimizer, buffer, mask,
                      dataclasses.replace(base, loss="unknown"))
        record("an unknown loss name is refused", False)
    except ValueError:
        record("an unknown loss name is refused", True)

    # A batch of different transitions, recomputed by hand. A generator with the buffer's seed
    # predicts the sampled indices. Identical transitions could not expose a broadcasting bug.
    transitions = [("a", "Ab", 2.0, "b", False), ("b", "Ae", 3.0, "d", False),
                   ("g", "Ah", 1.0, "h", True), ("c", "Af", 4.0, "f", False),
                   ("f", "Ag", 2.0, "g", False)]
    buffer = ReplayBuffer(8, np.random.default_rng(21))
    for s, a, c, s_next, done in transitions:
        buffer.add(STATE_INDEX[s], ACTION_INDEX[a], c, STATE_INDEX[s_next], done)
    cfg = dataclasses.replace(base, batch_size=16, gamma=0.5)
    online_q = all_state_q_values(online_net)
    target_q = all_state_q_values(target_net)
    squared_errors = []
    for i in np.random.default_rng(21).integers(0, len(transitions), size=cfg.batch_size):
        s, a, c, s_next, done = transitions[i]
        bootstrap = 0.0 if done else min(target_q[STATE_INDEX[s_next], ACTION_INDEX[label]]
                                         for label in ACTIONS[s_next])
        y = c + cfg.gamma * bootstrap
        squared_errors.append((y - online_q[STATE_INDEX[s], ACTION_INDEX[a]]) ** 2)
    expected = float(np.mean(squared_errors))
    loss = learning_step(online_net, target_net, optimizer, buffer, mask, cfg)
    record("the loss is the mean of (y - Q)^2 over a batch of different transitions, using "
           "batch_size and gamma from the config", abs(loss - expected) < 1e-4,
           f"loss {loss:.5f}, by hand {expected:.5f}")

    # The next gradient must be computed afresh, not added to the previous one.
    fresh_net = copy.deepcopy(online_net)
    fresh_net.zero_grad()
    state_before_sampling = buffer._rng.bit_generator.state
    states, actions, costs, next_states, dones = buffer.sample(cfg.batch_size)
    buffer._rng.bit_generator.state = state_before_sampling  # learning_step draws the same batch
    fresh_y = compute_targets(target_net, costs, next_states, dones, mask, cfg.gamma)
    fresh_q = fresh_net(encode(states)).gather(1, actions.unsqueeze(1)).squeeze(1)
    F.mse_loss(fresh_q, fresh_y).backward()
    learning_step(online_net, target_net, optimizer, buffer, mask, cfg)
    record("gradients do not accumulate across learning steps",
           all(torch.equal(p.grad, q.grad)
               for p, q in zip(online_net.parameters(), fresh_net.parameters())))


def check_parameter_count(record):
    count = count_parameters(QNetwork(_assignment_config().hidden))
    record("two hidden layers of 32 give 1575 trainable parameters",
           count == 1575 == (8 * 32 + 32) + (32 * 32 + 32) + (32 * 7 + 7), str(count))
    record("other hidden layouts follow the same formula",
           count_parameters(QNetwork((32,))) == (8 * 32 + 32) + (32 * 7 + 7)
           and count_parameters(QNetwork((8, 8))) == (8 * 8 + 8) + (8 * 8 + 8) + (8 * 7 + 7))
    layers = list(QNetwork((32, 32)).layers)
    record("layers are Linear(8,32), ReLU, Linear(32,32), ReLU, Linear(32,7)",
           [type(layer).__name__ for layer in layers]
           == ["Linear", "ReLU", "Linear", "ReLU", "Linear"]
           and (layers[0].in_features, layers[0].out_features) == (8, 32)
           and (layers[4].in_features, layers[4].out_features) == (32, 7))


def check_epsilon(record):
    cfg = _assignment_config()
    eps = np.array([epsilon_at(ep, cfg.eps_start, cfg.eps_end, cfg.eps_decay_episodes)
                    for ep in range(1, 5001)])
    record("epsilon is 1 at episode 1 and 0.05 from episode 4000 on",
           eps[0] == 1.0 and abs(eps[3999] - 0.05) < 1e-12
           and np.all(np.abs(eps[3999:] - 0.05) < 1e-12))
    formula = 1.0 - 0.95 * (np.arange(1, 4001) - 1) / 3999
    record("epsilon is linear: 1 - 0.95 * (ep - 1) / 3999",
           np.all(np.abs(eps[:4000] - formula) < 1e-12)
           and np.all(np.abs(np.diff(eps[:4000], 2)) < 1e-12))
    matched = tabular_settings(cfg)["tabular_matched"]
    a2_schedule = np.array([epsilon_schedule(ep, matched["eps_start"], matched["eps_min"],
                                             matched["decay_episodes"]) for ep in range(5000)])
    record("tabular_matched has the same epsilon as the neural agent in every episode",
           np.array_equal(a2_schedule, eps) and matched["decay_episodes"] == 3999)
    record("tabular_a2 keeps the decay submitted in Assignment 2",
           tabular_settings(cfg)["tabular_a2"]["decay_episodes"] == 2000)
    record("epsilon can be held constant", epsilon_at(777, 0.02, 0.02, 4000) == 0.02)


def check_replay_buffer(record):
    buffer = ReplayBuffer(5, np.random.default_rng(6))
    record("a new buffer is empty", len(buffer) == 0)
    for i in range(1, 4):
        buffer.add(i, i - 1, float(i), i, False)
    drawn = buffer.sample(2000)[0].tolist()
    record("a partly filled buffer samples only its filled part, with replacement",
           len(buffer) == 3 and set(drawn) == {1, 2, 3} and len(drawn) == 2000)
    for i in range(4, 8):
        buffer.add(i, i - 1, float(i), i, True)
    record("wrap-around overwrites the oldest transitions and holds the size at capacity",
           len(buffer) == 5 and buffer.states.tolist() == [6, 7, 3, 4, 5]
           and buffer.costs.tolist() == [6.0, 7.0, 3.0, 4.0, 5.0])
    states, actions, costs, next_states, dones = buffer.sample(5000)
    counts = np.bincount(states.numpy(), minlength=8)[3:]
    record("sampling is uniform over the stored transitions and keeps each tuple together",
           np.all(np.abs(counts / 5000 - 0.2) < 0.03) and bool(torch.equal(states - 1, actions))
           and bool(torch.equal(states, next_states)) and bool(torch.equal(states.float(), costs)))
    record("a batch is 1-D with the expected types",
           states.shape == (5000,) and states.dtype == torch.int64
           and costs.dtype == torch.float32 and dones.dtype == torch.bool)
    record("pair counts cover the filled part",
           int(buffer.pair_counts().sum()) == 5 and buffer.pair_counts()[6, 5] == 1)
    global_state = np.random.get_state()
    generator_state = buffer._rng.bit_generator.state
    buffer.sample(10)
    record("sampling draws from the buffer's own generator, not the global one",
           buffer._rng.bit_generator.state != generator_state
           and np.array_equal(np.random.get_state()[1], global_state[1])
           and np.random.get_state()[2] == global_state[2])


def check_learning_schedule(record):
    # M = 8 does not divide the 231 learning steps, so the run ends with the target lagging the
    # online network, and a log of the wrong network cannot pass for the right one.
    cfg = _short_config(hidden=(16,), episodes=70, warmup=50, target_update=8, capacity=1000,
                        lr=5e-4, eps_start=0.8, eps_end=0.1, eps_decay_episodes=30)
    real_learning_step = learning_step
    real_select_action = select_action
    expected_mask = _test_mask()
    sizes = []
    seen = {"wiring_ok": True}
    acting = []

    def learning_spy(online_net, target_net, optimizer, buffer, mask, spy_cfg):
        sizes.append(len(buffer))
        seen.update(buffer=buffer, capacity=buffer.capacity, lr=optimizer.param_groups[0]["lr"])
        seen["wiring_ok"] = (seen["wiring_ok"] and torch.equal(mask, expected_mask)
                             and spy_cfg == cfg)
        return real_learning_step(online_net, target_net, optimizer, buffer, mask, spy_cfg)

    def acting_spy(net, env, state, mask, eps, rng):
        acting.append((net, eps))
        return real_select_action(net, env, state, mask, eps, rng)

    run = _with_replaced_global("select_action", acting_spy, lambda: _with_replaced_global(
        "learning_step", learning_spy, lambda: train_neural(0, cfg)))
    record("no learning before warm-up, then exactly one learning step per environment step",
           sizes == list(range(50, 280 + 1)))
    record("the loop gives learning_step the environment's mask and the run's own config",
           seen["wiring_ok"])
    record("the loop acts with the online network and the epsilon of the current episode",
           len(acting) == 280 and all(net is run["online_net"] for net, _ in acting)
           and [eps for _, eps in acting[::4]] == [epsilon_at(ep, 0.8, 0.1, 30)
                                                    for ep in range(1, 71)])
    counts = (run["n_transitions"], run["n_learning_steps"], run["n_target_syncs"])
    record("a short run produces the expected counts",
           counts == expected_counts(cfg) == (280, 231, 28) and len(run["losses"]) == 231,
           str(counts))
    record("the assignment's configuration means 20000 transitions, 19801 learning steps, "
           "198 target syncs", expected_counts(_assignment_config()) == (20000, 19801, 198))
    record("the default configuration keeps the assignment's gamma = 1",
           NeuralConfig().gamma == 1.0)
    refused = 0
    for bad in [dict(capacity=100, warmup=200), dict(warmup=0)]:
        try:
            train_neural(0, _short_config(**bad))
        except ValueError:
            refused += 1
    record("a warm-up the buffer can never reach, or a warm-up below 1, is refused", refused == 2)
    record("training takes the network layout, capacity and learning rate from the config",
           count_parameters(run["online_net"]) == (8 * 16 + 16) + (16 * 7 + 7)
           and seen["capacity"] == 1000 and seen["lr"] == 5e-4)
    record("the logged epsilon follows the configured schedule, once per episode",
           np.array_equal(run["epsilons"], [epsilon_at(ep, 0.8, 0.1, 30) for ep in range(1, 71)]))
    record("the replay buffer holds feasible pairs only",
           np.all(run["replay_counts"][~run["mask"].numpy()] == 0))
    buffer = seen["buffer"]
    stored_ok = True
    for i in range(len(buffer)):
        s, a = STATES[buffer.states[i]], ACTION_LABELS[buffer.actions[i]]
        branches = [(STATE_INDEX[s_next], c) for s_next, _, c in TRANSITIONS[s][a]]
        stored_ok = (stored_ok and (buffer.next_states[i], buffer.costs[i]) in branches
                     and bool(buffer.dones[i]) == (STATES[buffer.next_states[i]] == TERMINAL))
    record("every stored tuple is a real transition: (s, a, cost, s', done) of a realized branch",
           stored_ok and len(buffer) == 280)
    record("logs have one row per episode and 10 Q columns",
           run["q_hist"].shape == (70, 10) and run["episode_costs"].shape == (70,)
           and run["epsilons"].shape == (70,)
           and set(np.unique(run["episode_costs"])) <= {8.0, 9.0, 11.0})
    final_q = all_state_q_values(run["online_net"])
    record("the last q_hist row holds the final network's values in the documented order",
           all(run["q_hist"][-1, i] == final_q[STATE_INDEX[s], ACTION_INDEX[a]]
               for i, (s, a) in enumerate(FEASIBLE_PAIRS)))


def check_training_streams(record):
    # With epsilon 1 and no learning (160 transitions never reach a warm-up of 1000), the episode
    # costs depend only on the environment and exploration streams, so they can be predicted
    # without the network.
    cfg = _short_config(episodes=40, warmup=1000, capacity=1000, eps_start=1.0, eps_end=1.0)
    costs_by_seed = {}
    predicted_ok = True
    for seed in [0, 7]:
        env_stream, explore_stream, _ = np.random.SeedSequence(seed).spawn(3)
        env = PathEnv(np.random.default_rng(env_stream))
        rng = np.random.default_rng(explore_stream)
        predicted = []
        for _ in range(cfg.episodes):
            state = env.reset()
            total = 0
            done = False
            while not done:
                labels = env.feasible_actions(state)
                if len(labels) > 1:
                    rng.random()  # the draw that is compared with epsilon
                label = labels[rng.integers(len(labels))] if len(labels) > 1 else labels[0]
                state, cost, done = env.step(label)
                total += cost
            predicted.append(total)
        costs_by_seed[seed] = train_neural(seed, cfg)["episode_costs"]
        predicted_ok = predicted_ok and np.array_equal(costs_by_seed[seed], predicted)
    record("environment noise and exploration come from SeedSequence(seed).spawn(3)", predicted_ok)
    record("different seeds see different environment noise and exploration",
           not np.array_equal(costs_by_seed[0], costs_by_seed[7]))

    real_learning_step = learning_step
    replay_states = []

    def spy(online_net, target_net, optimizer, buffer, mask, spy_cfg):
        replay_states.append(buffer._rng.bit_generator.state)
        return real_learning_step(online_net, target_net, optimizer, buffer, mask, spy_cfg)

    _with_replaced_global("learning_step", spy, lambda: train_neural(5, _short_config()))
    expected = np.random.default_rng(np.random.SeedSequence(5).spawn(3)[2]).bit_generator.state
    record("replay sampling starts from the third stream of SeedSequence(seed).spawn(3)",
           replay_states[0] == expected)

    # Weight initialization is the only use of the global generators, so all three must end
    # exactly where seeding plus one network construction leaves them.
    cfg = _short_config()
    train_neural(3, cfg)
    after_python = random.getstate()
    after_numpy = np.random.get_state()
    after_torch = torch.get_rng_state()
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    QNetwork(cfg.hidden)
    record("training consumes the global generators only for the weight initialization",
           random.getstate() == after_python
           and np.array_equal(np.random.get_state()[1], after_numpy[1])
           and np.random.get_state()[2] == after_numpy[2]
           and torch.equal(torch.get_rng_state(), after_torch))


def check_target_network(record):
    cfg = _short_config(episodes=30, warmup=20, target_update=10, capacity=1000)
    real_learning_step = learning_step
    log = []

    def spy(online_net, target_net, optimizer, buffer, mask, spy_cfg):
        online_before = weights_sha256(online_net)
        target_before = weights_sha256(target_net)
        loss = real_learning_step(online_net, target_net, optimizer, buffer, mask, spy_cfg)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        settings = optimizer.param_groups[0]
        log.append({
            "online_before": online_before,
            "target_before": target_before,
            "target_after": weights_sha256(target_net),
            "no_target_grad": all(p.grad is None and not p.requires_grad
                                  for p in target_net.parameters()),
            "online_grad": all(p.grad is not None for p in online_net.parameters()),
            "optimizer_online_only": optimized == {id(p) for p in online_net.parameters()},
            "adam_defaults": (isinstance(optimizer, torch.optim.Adam)
                              and settings["betas"] == (0.9, 0.999) and settings["eps"] == 1e-8
                              and settings["weight_decay"] == 0 and not settings["amsgrad"]),
            # Adam counts its own updates, so a re-created optimizer would start again at 1.
            "adam_step": float(optimizer.state[next(online_net.parameters())].get("step", -1)),
        })
        return loss

    run = _with_replaced_global("learning_step", spy, lambda: train_neural(0, cfg))
    record("target starts as an exact copy of the online network",
           log[0]["target_before"] == log[0]["online_before"])
    record("a learning step itself never changes the target network",
           all(e["target_before"] == e["target_after"] for e in log))
    # log[k] is learning step k + 1, so a sync after step k + 1 shows up in log[k + 1].
    changed_after = [k + 1 for k in range(len(log) - 1)
                     if log[k + 1]["target_before"] != log[k]["target_before"]]
    expected_syncs = [m for m in range(10, len(log), 10)]
    record("the target changes only every M learning steps", changed_after == expected_syncs,
           f"changes after steps {changed_after[:4]}...")
    record("a sync is a hard copy of the online weights",
           all(log[m]["target_before"] == log[m]["online_before"] for m in expected_syncs))
    record("run reports one sync per M learning steps", run["n_target_syncs"] == len(log) // 10)
    record("no gradient reaches the target network and its parameters are frozen",
           all(e["no_target_grad"] for e in log))
    record("Adam updates the online parameters only, and they receive gradient",
           all(e["optimizer_online_only"] and e["online_grad"] for e in log))
    record("one Adam with torch's default settings serves the whole run, never re-created",
           all(e["adam_defaults"] for e in log)
           and [e["adam_step"] for e in log] == list(range(1, len(log) + 1)))


def _model_based_calls():
    """One call to each function that reads the model or the reference numbers."""
    policy = {"a": "Ab", "b": "Ae", "c": "Af", "d": "Ag", "e": "Ag", "f": "Ag", "g": "Ah"}
    values = {"h": 0.0, "g": 1.0, "f": 3.0, "e": 4.0, "d": 4.0, "c": 7.4, "b": 6.2, "a": 8.44}
    return {
        "lookahead": lambda: lookahead(values, "a", "Ab"),
        "policy_evaluation": lambda: policy_evaluation(policy),
        "evaluate_general_policy": lambda: evaluate_general_policy(policy),
        "policy_improvement": lambda: policy_improvement(values),
        "run_policy_iteration": lambda: run_policy_iteration(),
        "reference_values": lambda: reference_values(),
        "nominal_path": lambda: nominal_path(policy),
        "loss_noise_floor": lambda: loss_noise_floor(np.ones((N_STATES, N_ACTIONS))),
    }


def _model_functions_not_blocked():
    """Names of the model-based functions that ran instead of raising RuntimeError."""
    not_blocked = []
    for name, call in _model_based_calls().items():
        try:
            call()
            not_blocked.append(name)
        except RuntimeError:
            pass
    return not_blocked


def check_model_guard(record):
    every_name = list(_model_based_calls())
    record("model-based functions work outside training",
           _model_functions_not_blocked() == every_name)

    seen = {}
    real_learning_step = learning_step
    real_sampler = step

    def neural_spy(*args):
        seen.setdefault("neural", _model_functions_not_blocked())
        return real_learning_step(*args)

    def tabular_spy(rng, s, a):
        seen.setdefault("tabular", _model_functions_not_blocked())
        return real_sampler(rng, s, a)

    def crash(*args):
        raise ArithmeticError("deliberate failure inside training")

    _with_replaced_global("learning_step", neural_spy, lambda: train_neural(0, _short_config()))
    record("every model-based function raises RuntimeError during train_neural",
           seen["neural"] == [], ", ".join(seen["neural"]))
    _with_replaced_global("step", tabular_spy, lambda: run_q_learning(0, n_episodes=2))
    record("every model-based function raises RuntimeError during run_q_learning",
           seen["tabular"] == [], ", ".join(seen["tabular"]))
    record("the guard is released after training", not _training_active)
    try:
        _with_replaced_global("learning_step", crash, lambda: train_neural(0, _short_config()))
    except ArithmeticError:
        pass
    record("the guard is released even when training raises (try/finally)", not _training_active)


def check_source_inspection(record):
    model_names = ["TRANSITIONS", "lookahead", "policy_evaluation", "evaluate_general_policy",
                   "policy_improvement", "run_policy_iteration", "reference_values",
                   "nominal_path", "loss_noise_floor", "q_star", "v_star", "pi_star"]

    def forbidden_in(objects, names):
        found = []
        for obj in objects:
            source = inspect.getsource(obj)
            found += [f"{obj.__name__}:{name}" for name in names
                      if re.search(rf"\b{name}\b", source)]
        return found

    neural_learner = [train_neural, learning_step, compute_targets, select_action, masked_argmin,
                      encode, epsilon_at, feasibility_mask, all_state_q_values, QNetwork,
                      ReplayBuffer]
    # FEASIBLE_PAIRS is built from ACTIONS, so it is just as forbidden: the mask must come from
    # env.feasible_actions.
    found = forbidden_in(neural_learner, model_names + ["ACTIONS", "FEASIBLE_PAIRS",
                                                        "run_q_learning"])
    # A call of the bare sampler (env.step and optimizer.step are preceded by a dot), or a look
    # at the environment's private state: the learner may use PathEnv's public API only.
    found += [obj.__name__ + ":sampler" for obj in neural_learner
              if re.search(r"(?<![\w.])step\(", inspect.getsource(obj))]
    found += [obj.__name__ + ":env internals" for obj in neural_learner
              if re.search(r"\benv\._", inspect.getsource(obj))]
    record("the neural learner names no model, reference or tabular object, no sampler and no "
           "private state of the environment", not found, ", ".join(found))
    found = forbidden_in([PathEnv], model_names + ["train_neural", "run_q_learning"])
    record("PathEnv goes through the sampler and names no model table directly", not found,
           ", ".join(found))
    found = forbidden_in([run_q_learning], model_names + ["train_neural"])
    record("the tabular learner names no model or reference object", not found, ", ".join(found))

    readers = {name for name, obj in globals().items()
               if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == __name__
               and not name.startswith("check_")
               and re.search(r"\bTRANSITIONS\b", inspect.getsource(obj))}
    record("TRANSITIONS is read only by the sampler and by guarded post-hoc functions",
           readers == {"step", "lookahead", "nominal_path", "loss_noise_floor"},
           ", ".join(sorted(readers)))

    # Runtime proof: the table is emptied and refilled only inside the sampler, so any other
    # reader during training finds nothing, even through another name for the same dict. The
    # sampler counts its calls, so a caller other than the environment would show up too.
    entries = dict(TRANSITIONS)
    real_sampler = step
    n_calls = [0]

    def sampler_only(rng, s, a):
        n_calls[0] += 1
        TRANSITIONS.update(entries)
        try:
            return real_sampler(rng, s, a)
        finally:
            TRANSITIONS.clear()

    cfg = _short_config()
    normal = train_neural(0, cfg)
    TRANSITIONS.clear()
    globals()["step"] = sampler_only
    try:
        hidden = train_neural(0, cfg)
        neural_calls = n_calls[0]
        run_q_learning(0, n_episodes=20)
        tabular_calls = n_calls[0] - neural_calls
    finally:
        TRANSITIONS.update(entries)
        globals()["step"] = real_sampler
    record("training gives the same weights with the table visible only inside the sampler",
           hidden["weights_sha256"] == normal["weights_sha256"])
    record("during training the sampler runs once per environment step and for nothing else",
           neural_calls == hidden["n_transitions"] and tabular_calls == 4 * 20,
           f"{neural_calls} and {tabular_calls} calls")


def check_policy_iteration(record):
    policy, values = run_policy_iteration()
    record("policy iteration gives V*(a) = 8.44 with policy Ab, Ae, Af",
           abs(values["a"] - 8.44) < 1e-9 and policy == {"a": "Ab", "b": "Ae", "c": "Af"},
           f"V*(a) = {values['a']:.4f}")
    # Numbers from results/pi_trace.txt of Assignment 2.
    expected_v = {"a": 8.44, "b": 6.2, "c": 7.4, "d": 4.0, "e": 4.0, "f": 3.0, "g": 1.0, "h": 0.0}
    expected_q = {"a|Ab": 8.44, "a|Ac": 9.16, "b|Ad": 6.8, "b|Ae": 6.2, "c|Ae": 8.6, "c|Af": 7.4,
                  "d|Ag": 4.0, "e|Ag": 4.0, "f|Ag": 3.0, "g|Ah": 1.0}
    reference = reference_values()
    record("V* matches pi_trace.txt of Assignment 2",
           all(abs(reference["v_star"][s] - v) < 1e-9 for s, v in expected_v.items()))
    record("Q* matches pi_trace.txt of Assignment 2",
           sorted(reference["q_star"]) == sorted(expected_q)
           and all(abs(reference["q_star"][k] - v) < 1e-9 for k, v in expected_q.items()))
    pi0 = {"a": {"Ab": 0.60, "Ac": 0.40}, "b": {"Ad": 0.70, "Ae": 0.30},
           "c": {"Ae": 0.50, "Af": 0.50}}
    first_sweep = evaluate_general_policy(pi0)
    record("sweep 1 of the stochastic initial policy matches pi_trace.txt",
           abs(first_sweep["a"] - 9.2272) < 1e-9 and abs(first_sweep["b"] - 6.62) < 1e-9
           and abs(first_sweep["c"] - 8.0) < 1e-9)
    record("the reference policy covers all 7 nonterminal states",
           reference["pi_star"] == {"a": "Ab", "b": "Ae", "c": "Af", "d": "Ag", "e": "Ag",
                                    "f": "Ag", "g": "Ah"})


def check_tabular_baseline(record):
    # Printed once by the real Assignment 2 code: run_q_learning(4, n_episodes=300).
    a2_q = {"a": {"Ab": 8.446609969130494, "Ac": 9.04878929635314},
            "b": {"Ad": 6.666470694975773, "Ae": 6.182063719274601},
            "c": {"Ae": 8.610620819616637, "Af": 7.380496467352637},
            "d": {"Ag": 3.998860221029632},
            "e": {"Ag": 3.999998499064422},
            "f": {"Ag": 2.999179038881048},
            "g": {"Ah": 0.9999999999999813}}
    run = run_q_learning(4, n_episodes=300)
    values, counts = np.unique(run["episode_costs"], return_counts=True)
    record("the tabular Q table equals Assignment 2 exactly (seed 4, 300 episodes)",
           run["Q"] == a2_q)
    record("the tabular episode costs equal Assignment 2 (sum and counts)",
           run["episode_costs"].sum() == 2749.0
           and dict(zip(values.tolist(), counts.tolist())) == {8.0: 79, 9.0: 157, 11.0: 64})
    final_q = [run["Q"][s][a] for s, a in FEASIBLE_PAIRS]
    record("the per-episode Q snapshot ends at the final table",
           run["q_hist"].shape == (300, 10) and np.array_equal(run["q_hist"][-1], final_q))
    record("the tabular learner makes 4 updates per episode", run["n_transitions"] == 1200)

    settings = tabular_settings(_assignment_config())
    record("the reported baselines use the settings of Assignment 2 and of the matched schedule",
           settings == {"tabular_a2": dict(n_episodes=5000, alpha=0.1, eps_start=1.0, eps_min=0.05,
                                           decay_episodes=2000),
                        "tabular_matched": dict(n_episodes=5000, alpha=0.1, eps_start=1.0,
                                                eps_min=0.05, decay_episodes=3999)})
    # Printed once by the real Assignment 2 code for seeds 0-4 and its defaults: the first 16 hex
    # digits of the SHA-256 of the final Q (float64, order of FEASIBLE_PAIRS) and the episode costs.
    a2_fingerprints = ["41cb2bb33fc905a6", "873a6fe687ffe693", "a13fa1df02fc774b",
                       "3a13834a00982b9b", "b947c9dd465b5adb"]
    fingerprints = []
    for seed in range(5):
        a2_run = run_q_learning(seed, **settings["tabular_a2"])
        final_q = np.array([a2_run["Q"][s][a] for s, a in FEASIBLE_PAIRS])
        digest = hashlib.sha256(final_q.tobytes() + a2_run["episode_costs"].tobytes())
        fingerprints.append(digest.hexdigest()[:16])
    record("tabular_a2 equals Assignment 2 exactly on seeds 0-4 over all 5000 episodes",
           fingerprints == a2_fingerprints, ", ".join(fingerprints))


def check_reproducibility(record):
    cfg = _short_config(episodes=100, warmup=50, target_update=20, eps_decay_episodes=80)
    first = train_neural(0, cfg)
    again = train_neural(0, cfg)
    other = train_neural(1, cfg)
    record("same seed twice gives an identical final-weights SHA-256",
           first["weights_sha256"] == again["weights_sha256"], first["weights_sha256"][:12])
    record("same seed twice gives identical logs",
           np.array_equal(first["losses"], again["losses"])
           and np.array_equal(first["q_hist"], again["q_hist"])
           and np.array_equal(first["episode_costs"], again["episode_costs"]))
    record("different seeds give different weights",
           first["weights_sha256"] != other["weights_sha256"])

    train_neural(3, cfg)
    random.random()
    np.random.random(1000)
    torch.rand(1000)
    polluted = train_neural(0, cfg)
    record("another seed first plus polluted global generators do not change a seed's hash",
           polluted["weights_sha256"] == first["weights_sha256"])

    evaluate_policy(neural_actor(other["online_net"], other["mask"], evaluation_tie_stream(1)),
                    1, n_episodes=50)
    after_evaluation = train_neural(0, cfg)
    record("evaluating between runs does not change a seed's hash",
           after_evaluation["weights_sha256"] == first["weights_sha256"])
    weights_before = weights_sha256(other["online_net"])
    evaluate_policy(neural_actor(other["online_net"], other["mask"], evaluation_tie_stream(1)),
                    1, n_episodes=50)
    record("evaluation leaves the network untouched",
           weights_sha256(other["online_net"]) == weights_before)

    # Sweep rows must not depend on the process or on what it ran before.
    def row_without_clock(row):
        return {key: value for key, value in row.items() if key != "wall_clock_s"}

    task = ("check", "neural", cfg, 100)
    row = row_without_clock(run_sweep_task(task))
    run_sweep_task(("check", "tabular", dict(n_episodes=20), 101))
    train_neural(3, cfg)
    record("a sweep task gives the same row after other work in the same process",
           row_without_clock(run_sweep_task(task)) == row)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        worker_row = pool.submit(run_sweep_task, task).result()
    record("a spawned worker gives the same sweep row as this process",
           row_without_clock(worker_row) == row)


def check_sweep_definition(record):
    variants = sweep_variants()
    neural = {name: params for name, method, params in variants if method == "neural"}
    tabular = {name: params for name, method, params in variants if method == "tabular"}
    record("the sweep has 24 neural and 10 tabular variants with unique names",
           len(neural) == 24 and len(tabular) == 10
           and len(variants) == len(neural) + len(tabular))
    record("the sweep's seeds are held out from the reported seeds 0-4",
           all(sweep_seeds(k) == list(range(100, 100 + k)) for k in [1, 10, 50]))
    record("the sweep's Assignment 2 baselines keep the settings of Assignment 2",
           tabular["a2_baseline"] == dict(n_episodes=5000, alpha=0.1, decay_episodes=2000)
           and tabular["a2_aggressive"] == dict(n_episodes=5000, alpha=0.99, eps_start=0.02,
                                                eps_min=0.02, decay_episodes=1))
    base = neural["baseline"]
    differences = {name: {f.name for f in dataclasses.fields(cfg)
                          if getattr(cfg, f.name) != getattr(base, f.name)}
                   for name, cfg in neural.items()}
    expected = {"lr": {"lr"}, "M": {"target_update"}, "batch": {"batch_size"}, "hidden": {"hidden"},
                "capacity": {"capacity"}, "huber": {"loss"}, "eps_decay": {"eps_decay_episodes"},
                "eps_const": {"eps_start", "eps_end"},
                "episodes": {"episodes", "eps_decay_episodes"},
                "no_replay": {"capacity", "warmup", "batch_size"}}
    # At most, not exactly: a variant equals the baseline if the defaults later move onto it.
    record("every neural variant changes at most one factor of the baseline",
           all(differences[name] <= expected[name.split("=")[0]]
               for name in neural if name != "baseline"))
    no_replay = neural["no_replay"]
    record("no_replay trains on the newest transition only",
           (no_replay.capacity, no_replay.warmup, no_replay.batch_size) == (1, 1, 1))
    record("the matched tabular variants use the neural epsilon schedule",
           tabular["alpha=0.1"]["decay_episodes"] + 1 == base.eps_decay_episodes
           and all(tabular[name]["decay_episodes"] + 1 == neural[name].eps_decay_episodes
                   for name in ["episodes=1000", "episodes=2000"]))
    short = {(name, method): params for name, method, params in sweep_variants(100)}
    record("an --episodes override sets every variant's length and keeps schedules proportional",
           all((p.episodes if method == "neural" else p["n_episodes"]) == 100
               for (_, method), p in short.items())
           and short[("episodes=1000", "neural")].eps_decay_episodes == 80
           and short[("eps_decay=2000", "neural")].eps_decay_episodes
           == round(2000 * 100 / neural["eps_decay=2000"].episodes)
           and short[("episodes=1000", "tabular")]["decay_episodes"] == 79)

    def row(seed, stable, gap=0.0):
        return {"variant": "v", "method": "neural", "seed": seed, "is_optimal": gap == 0.0,
                "gap": gap, "q_rmse": 0.1 * seed, "optimal_frac_last_1000": 1.0,
                "switches_last_1000": seed, "first_stable_episode": stable, "wall_clock_s": 1.0}

    summary = summarize_sweep([row(1, 10), row(2, 30), row(3, None, 0.6)])[0]
    never = summarize_sweep([row(1, 10), row(2, None, 0.6), row(3, None, 0.6)])[0]
    record("the sweep summary counts optimal seeds, averages per variant and treats an unstable "
           "run as never stable",
           summary["n_seeds"] == 3 and summary["seeds_optimal"] == 2
           and abs(summary["mean_gap"] - 0.2) < 1e-9 and summary["max_gap"] == 0.6
           and abs(summary["mean_q_rmse"] - 0.2) < 1e-9 and summary["mean_switches"] == 2.0
           and summary["median_first_stable"] == 30.0 and never["median_first_stable"] == "")


def check_evaluation(record):
    reference = reference_values()
    pi_star = reference["pi_star"]
    stats = evaluate_policy(table_actor(pi_star), 0)
    totals = np.concatenate([np.full(c, float(v)) for v, c in stats["cost_counts"].items()])
    record("evaluation runs 1000 episodes, all reach h, costs in {8, 9, 11}",
           stats["n_episodes"] == 1000 and len(totals) == 1000 and stats["pct_reach_h"] == 100.0
           and set(stats["cost_counts"]) <= {"8", "9", "11"})
    record("the statistics are mean, std with ddof 1, min, max and the normal 95% half width",
           abs(stats["mean"] - totals.mean()) < 1e-12
           and abs(stats["std"] - totals.std(ddof=1)) < 1e-12
           and stats["min"] == totals.min() and stats["max"] == totals.max()
           and abs(stats["ci95_half_width"] - 1.96 * stats["std"] / np.sqrt(1000)) < 1e-12)
    record("the evaluated mean of pi* agrees with the exact V*(a)",
           abs(stats["mean"] - reference["v_star"]["a"]) < 3 * stats["ci95_half_width"],
           f"mean {stats['mean']:.4f}, half width {stats['ci95_half_width']:.4f}")

    # Common random numbers: a network with the same greedy actions sees the same episodes.
    mask = _test_mask()
    table = torch.full((N_STATES, N_ACTIONS), -1e6)  # infeasible outputs are the lowest
    for s, a in FEASIBLE_PAIRS:
        table[STATE_INDEX[s], ACTION_INDEX[a]] = 0.0 if pi_star[s] == a else 1.0
    stub_actor = neural_actor(_stub_network(table), mask, evaluation_tie_stream(0))
    neural_stats = evaluate_policy(stub_actor, 0)
    record("neural and tabular actors of one policy give identical results on a seed's stream",
           neural_stats == stats)
    record("a different seed gets a different evaluation stream",
           evaluate_policy(table_actor(pi_star), 1)["cost_counts"] != stats["cost_counts"])
    training_streams = np.random.SeedSequence(0).spawn(3)
    first_values = [np.random.default_rng(stream).random() for stream in training_streams]
    record("the evaluation stream differs from the training streams of the seed",
           evaluation_stream(0).random() not in first_values
           and np.random.default_rng(0).random() != evaluation_stream(0).random())
    guarded = evaluate_policy(table_actor(pi_star), 0, n_episodes=10, max_steps=2)
    record("the step guard makes nontermination visible instead of hanging",
           guarded["pct_reach_h"] == 0.0)

    # Replayed by hand on the stream the specification names, episode for episode.
    env = PathEnv(np.random.default_rng(np.random.SeedSequence([598003, 0])))
    replayed = []
    for _ in range(1000):
        state, total, done = env.reset(), 0, False
        while not done:
            state, cost, done = env.step(pi_star[state])
            total += cost
        replayed.append(total)
    values, counts = np.unique(replayed, return_counts=True)
    record("evaluate_policy draws its noise from default_rng(SeedSequence([598003, seed]))",
           stats["cost_counts"] == {str(int(v)): int(c) for v, c in zip(values, counts)}
           and stats["mean"] == np.mean(replayed))


def check_post_hoc_metrics(record):
    ends, means = rolling_mean(np.arange(1.0, 301.0), 100)
    record("the rolling mean at episode k averages episodes k-99..k",
           ends[0] == 100 and ends[-1] == 300 and abs(means[0] - 50.5) < 1e-9
           and abs(means[-1] - 250.5) < 1e-9 and len(ends) == len(means) == 201)
    record("the rolling mean of a too short series is empty",
           len(rolling_mean(np.ones(5), 100)[0]) == 0)

    reference = reference_values()
    q_star = np.array([reference["q_star"][f"{s}|{a}"] for s, a in FEASIBLE_PAIRS])
    wrong = q_star.copy()
    wrong[[0, 1]] = wrong[[1, 0]]  # prefers Ac at a

    q_hist = np.tile(q_star, (1500, 1))
    q_hist[:300] = wrong
    q_hist[800] = wrong
    metrics = stability_metrics(q_hist, reference)
    record("greedy_ok and first_stable_episode on a synthetic history",
           metrics["first_stable_episode"] == 802 and not metrics["greedy_ok"][800]
           and metrics["greedy_ok"][801] and metrics["q_rmse"] < 1e-12)
    record("optimal_frac_last_1000 and switches_last_1000 on a synthetic history",
           abs(metrics["optimal_frac_last_1000"] - 0.999) < 1e-12
           and metrics["switches_last_1000"] == 2)
    record("first_stable_episode is 1 when the policy is always optimal",
           stability_metrics(np.tile(q_star, (20, 1)), reference)["first_stable_episode"] == 1)
    q_hist[-1] = wrong
    record("first_stable_episode is null when the run ends non-optimal",
           stability_metrics(q_hist, reference)["first_stable_episode"] is None)
    tied = q_star.copy()
    tied[1] = tied[0]
    record("an exact tie does not count as the optimal action",
           not stability_metrics(np.tile(tied, (3, 1)), reference)["greedy_ok"].any())
    short = stability_metrics(np.tile(q_star, (10, 1)), reference)
    record("the metrics degrade gracefully when a run has fewer than 1000 episodes",
           short["optimal_frac_last_1000"] == 1.0 and short["switches_last_1000"] == 0)
    one = stability_metrics(np.tile(q_star, (1, 1)), reference)
    record("the metrics work for a single episode",
           one["first_stable_episode"] == 1 and one["switches_last_1000"] == 0)
    swapped_b = q_star.copy()
    swapped_b[[2, 3]] = swapped_b[[3, 2]]  # b|Ad and b|Ae swapped: errors of 0.6 in 2 of 10 values
    record("q_rmse is the root of the mean squared error over the 10 values",
           abs(stability_metrics(np.tile(swapped_b, (2, 1)), reference)["q_rmse"]
               - np.sqrt(2 * 0.6 ** 2 / 10)) < 1e-12)

    counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    counts[STATE_INDEX["a"], ACTION_INDEX["Ab"]] = 1
    only_a = loss_noise_floor(counts)
    counts[STATE_INDEX["g"], ACTION_INDEX["Ah"]] = 1
    half = loss_noise_floor(counts)
    counts[:] = 0
    counts[STATE_INDEX["c"], ACTION_INDEX["Af"]] = 3
    only_c = loss_noise_floor(counts)
    record("the noise floor matches hand-computed variances",
           abs(only_a - 0.2304) < 1e-9 and abs(half - 0.1152) < 1e-9 and abs(only_c - 0.64) < 1e-9)

    forced = {"d": "Ag", "e": "Ag", "f": "Ag", "g": "Ah"}
    record("nominal paths and costs follow the intended branches",
           nominal_path({"a": "Ab", "b": "Ae", **forced}) == ("a->b->e->g->h", 8)
           and nominal_path({"a": "Ab", "b": "Ad", **forced}) == ("a->b->d->g->h", 9)
           and nominal_path({"a": "Ac", "c": "Ae", **forced}) == ("a->c->e->g->h", 11)
           and nominal_path({"a": "Ac", "c": "Af", **forced}) == ("a->c->f->g->h", 9))

    matrix = table_to_matrix(run_q_learning(1, n_episodes=200)["Q"])
    policy = greedy_policy(matrix, _test_mask(), evaluation_tie_stream(0))
    record("greedy_policy gives a feasible action at all 7 nonterminal states",
           sorted(policy) == list("abcdefg") and all(policy[s] in ACTIONS[s] for s in policy))
    star = np.full((N_STATES, N_ACTIONS), -1e9)  # infeasible cells are the most attractive
    for s, a in FEASIBLE_PAIRS:
        star[STATE_INDEX[s], ACTION_INDEX[a]] = reference["q_star"][f"{s}|{a}"]
    record("greedy_policy of Q* is pi*, never a more attractive infeasible action",
           greedy_policy(star, _test_mask(), evaluation_tie_stream(0)) == reference["pi_star"])


def check_main_run(record):
    """One short run_seed into a scratch folder; its records, logs, table and figure are compared
    with the functions they come from. After 120 episodes both tabular policies of seed 4 are
    still non-optimal, so a wrong policy, gap or evaluation cannot hide behind pi*."""
    cfg = _short_config(episodes=120, eps_decay_episodes=96)
    seed = 4
    reference = reference_values()
    with tempfile.TemporaryDirectory() as folder:
        for name in ["training_log_seed9.npz", "fig_stability.png", "policies.csv", "checks.txt",
                     "sweep_runs.csv", "notes.txt"]:
            open(os.path.join(folder, name), "w", encoding="utf-8").close()
        remove_old_outputs(folder)
        record("a new main run deletes only what an earlier main run wrote",
               sorted(os.listdir(folder)) == ["checks.txt", "notes.txt", "sweep_runs.csv"])
        records = run_seed(seed, cfg, folder)
        write_tables({method: {str(seed): records[method]} for method in METHODS}, folder)
        with open(os.path.join(folder, "five_seed_evaluation.csv"), encoding="utf-8") as f:
            table = {row["method"]: row for row in csv.DictReader(f)}
        with np.load(os.path.join(folder, f"training_log_seed{seed}.npz")) as log:
            logs = {"neural": {key: log[key] for key in log.files}}
        with np.load(os.path.join(folder, f"tabular_log_seed{seed}.npz")) as log:
            for method, prefix in [("tabular_a2", "a2_"), ("tabular_matched", "matched_")]:
                logs[method] = {key[len(prefix):]: log[key] for key in log.files
                                if key.startswith(prefix)}

    consistent = True
    for method, r in records.items():
        log = logs[method]
        exact = policy_evaluation(r["policy"])["a"]
        consistent = (consistent and r["exact_expected_cost"] == exact
                      and r["gap"] == exact - reference["v_star"]["a"]
                      and r["is_optimal"] == (abs(r["gap"]) < 1e-6)
                      and list(r["q_final"].values()) == log["q_hist"][-1].tolist()
                      and r["q_rmse"] == stability_metrics(log["q_hist"], reference)["q_rmse"]
                      and r["rolling_cost_last_100"] == np.mean(log["episode_costs"][-100:])
                      and float(table[method]["std_cost"]) == round(r["eval"]["std"], 6))
    record("every method's record agrees with its saved log, the exact evaluation and the table",
           consistent)
    tabular = ["tabular_a2", "tabular_matched"]
    record("the tabular records evaluate their own (here non-optimal) policy on the seed's stream",
           not any(records[m]["is_optimal"] for m in tabular)
           and all(records[m]["eval"] == evaluate_policy(table_actor(records[m]["policy"]), seed)
                   for m in tabular))
    a2_run = run_q_learning(seed, **tabular_settings(cfg)["tabular_a2"])
    record("each baseline's log is saved under its own name: matched has the neural epsilon",
           np.array_equal(logs["tabular_matched"]["epsilons"], logs["neural"]["epsilons"])
           and np.array_equal(logs["tabular_a2"]["q_hist"][-1],
                              [a2_run["Q"][s][a] for s, a in FEASIBLE_PAIRS]))

    plotted = {}

    def keep_lines(fig, ax, out_dir, name):  # stands in for finish_figure, so nothing is saved
        plotted[name] = [line.get_xdata() for line in ax.get_lines()]
        plt.close(fig)

    _with_replaced_global("finish_figure", keep_lines, lambda: plot_seed_cost(
        logs["neural"]["episode_costs"], reference["v_star"]["a"], None, "cost"))
    rolling_x = plotted["cost"][1]  # the lines are the raw costs, the rolling mean, V*(a)
    record("the plotted rolling mean at episode k averages episodes k-99..k",
           rolling_x[0] == EPISODE_WINDOW and rolling_x[-1] == cfg.episodes)

    # Series as long as their rolling windows, whose rolling means are a single point. A line needs
    # two points to be drawn, and every line of these figures gets a legend entry, so a shorter
    # line would be a legend entry with nothing drawn.
    def plot_short_series():
        plot_seed_cost(np.full(EPISODE_WINDOW, 9.0), reference["v_star"]["a"], None, "short cost")
        plot_seed_loss(np.full(LOSS_WINDOW, 0.5), 0.1, "mse", None, "short loss")

    _with_replaced_global("finish_figure", keep_lines, plot_short_series)
    record("a series no longer than its rolling window lists no undrawn line in the legend",
           all(len(x) > 1 for name in ["short cost", "short loss"] for x in plotted[name]))


CHECKS = [check_transition_table, check_feasible_sets, check_empirical_branches, check_episodes,
          check_env_errors, check_mask_and_argmin, check_action_selection, check_compute_targets,
          check_learning_step, check_parameter_count, check_epsilon, check_replay_buffer,
          check_learning_schedule, check_training_streams, check_target_network, check_model_guard,
          check_source_inspection, check_policy_iteration, check_tabular_baseline,
          check_reproducibility, check_sweep_definition, check_evaluation, check_post_hoc_metrics,
          check_main_run]


def run_checks(out_dir):
    """Run every check, print PASS or FAIL per check, write checks.txt, exit non-zero on failure."""
    torch.set_num_threads(1)  # the setting of every training run, so comparisons are exact
    lines = []
    n_failed = 0
    start_time = time.perf_counter()

    def record(name, passed, detail=""):
        nonlocal n_failed
        passed = bool(passed)
        n_failed += 0 if passed else 1
        line = ("PASS  " if passed else "FAIL  ") + name + (f"  [{detail}]" if detail else "")
        lines.append(line)
        print(line, flush=True)

    for check in CHECKS:
        try:
            check(record)
        except Exception as exc:  # a crash in one check must not hide the others
            record(f"{check.__name__} runs to completion", False, repr(exc))

    summary = (f"{len(lines)} checks, {n_failed} failed, "
               f"{time.perf_counter() - start_time:.1f} s")
    lines.append(summary)
    print(summary, flush=True)
    with open(os.path.join(out_dir, "checks.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if n_failed:
        print("Self-checks FAILED, aborting.", flush=True)
        sys.exit(1)


# --------------------------------------------------------------------------
# 11. Sensitivity sweep
# --------------------------------------------------------------------------
# Held-out seeds: tuning never touches the seeds 0-4 reported in the main table.
SWEEP_FIRST_SEED = 100


def sweep_seeds(n_seeds):
    """The seeds of the sweep: 100, 101, ..."""
    return [SWEEP_FIRST_SEED + i for i in range(n_seeds)]


def sweep_variants(episodes_override=None):
    """One-factor-at-a-time variants of the baseline, as (variant, method, parameters).
    An episodes override keeps every schedule proportional; it exists for smoke tests only."""
    replace = dataclasses.replace
    base = NeuralConfig()
    neural = [
        ("baseline", base),
        ("lr=1e-4", replace(base, lr=1e-4)),
        ("lr=3e-4", replace(base, lr=3e-4)),
        ("lr=3e-3", replace(base, lr=3e-3)),
        ("lr=1e-2", replace(base, lr=1e-2)),
        ("M=1", replace(base, target_update=1)),
        ("M=10", replace(base, target_update=10)),
        ("M=500", replace(base, target_update=500)),
        ("M=2000", replace(base, target_update=2000)),
        ("batch=8", replace(base, batch_size=8)),
        ("batch=64", replace(base, batch_size=64)),
        ("batch=128", replace(base, batch_size=128)),
        ("hidden=8x8", replace(base, hidden=(8, 8))),
        ("hidden=32", replace(base, hidden=(32,))),
        ("hidden=128x128", replace(base, hidden=(128, 128))),
        ("capacity=500", replace(base, capacity=500)),
        ("capacity=2000", replace(base, capacity=2000)),
        ("capacity=20000", replace(base, capacity=20000)),
        ("no_replay", replace(base, capacity=1, warmup=1, batch_size=1)),
        ("huber", replace(base, loss="huber")),
        ("eps_decay=2000", replace(base, eps_decay_episodes=2000)),
        ("eps_const=0.02", replace(base, eps_start=0.02, eps_end=0.02)),
        ("episodes=1000", replace(base, episodes=1000, eps_decay_episodes=800)),
        ("episodes=2000", replace(base, episodes=2000, eps_decay_episodes=1600)),
    ]
    # The Assignment 2 schedule is zero-indexed, so decay D here equals decay D + 1 above.
    matched = dict(n_episodes=base.episodes, alpha=0.1, eps_start=base.eps_start,
                   eps_min=base.eps_end, decay_episodes=base.eps_decay_episodes - 1)
    tabular = [(f"alpha={alpha}", dict(matched, alpha=alpha))
               for alpha in [0.01, 0.05, 0.1, 0.3, 0.5, 0.99]]
    tabular += [
        ("a2_baseline", dict(n_episodes=5000, alpha=0.1, decay_episodes=2000)),
        ("a2_aggressive", dict(n_episodes=5000, alpha=0.99, eps_start=0.02, eps_min=0.02,
                               decay_episodes=1)),
        ("episodes=1000", dict(matched, n_episodes=1000, decay_episodes=799)),
        ("episodes=2000", dict(matched, n_episodes=2000, decay_episodes=1599)),
    ]
    if episodes_override is not None:
        n = episodes_override
        neural = [(name, replace(cfg, episodes=n, eps_decay_episodes=max(
            1, round(cfg.eps_decay_episodes * n / cfg.episodes)))) for name, cfg in neural]
        tabular = [(name, dict(params, n_episodes=n, decay_episodes=max(
            1, round((params["decay_episodes"] + 1) * n / params["n_episodes"]) - 1)))
            for name, params in tabular]
    return ([(name, "neural", cfg) for name, cfg in neural]
            + [(name, "tabular", params) for name, params in tabular])


def run_sweep_task(task):
    """Train one variant on one seed. Must stay a module-level function so that spawned workers
    can import it. Everything is seeded inside the training function, so the result depends only
    on this task, not on the worker that runs it or on what that worker ran before."""
    variant, method, params, seed = task
    mask = feasibility_mask(PathEnv(np.random.default_rng(0)))
    tie_rng = evaluation_tie_stream(seed)
    if method == "neural":
        run = train_neural(seed, params)
        policy = greedy_policy(all_state_q_values(run["online_net"]), mask, tie_rng)
    else:
        run = run_q_learning(seed, **params)
        policy = greedy_policy(table_to_matrix(run["Q"]), mask, tie_rng)
    reference = reference_values()  # only now that training has finished
    metrics = stability_metrics(run["q_hist"], reference)
    exact_cost = policy_evaluation(policy)["a"]
    gap = exact_cost - reference["v_star"]["a"]
    return {
        "variant": variant,
        "method": method,
        "seed": seed,
        "is_optimal": bool(abs(gap) < 1e-6),
        "gap": round(gap, 6),
        "exact_cost": round(exact_cost, 6),
        "q_rmse": round(metrics["q_rmse"], 8),
        "optimal_frac_last_1000": metrics["optimal_frac_last_1000"],
        "switches_last_1000": metrics["switches_last_1000"],
        "first_stable_episode": metrics["first_stable_episode"],
        "wall_clock_s": round(run["wall_clock_s"], 3),
    }


def summarize_sweep(rows):
    """One row per variant, in the order of the variant list."""
    groups = {}
    for row in rows:
        groups.setdefault((row["variant"], row["method"]), []).append(row)
    summary = []
    for (variant, method), group in groups.items():
        # A run that never stabilizes counts as later than any episode, so the median stays
        # honest; a blank median means at least half of the seeds did not end stable.
        stable = [np.inf if r["first_stable_episode"] is None else r["first_stable_episode"]
                  for r in group]
        median_stable = float(np.median(stable))
        summary.append({
            "variant": variant,
            "method": method,
            "n_seeds": len(group),
            "seeds_optimal": sum(r["is_optimal"] for r in group),
            "mean_gap": round(float(np.mean([r["gap"] for r in group])), 6),
            "max_gap": round(float(np.max([r["gap"] for r in group])), 6),
            "mean_q_rmse": round(float(np.mean([r["q_rmse"] for r in group])), 8),
            "mean_optimal_frac": round(float(np.mean([r["optimal_frac_last_1000"]
                                                      for r in group])), 6),
            "mean_switches": round(float(np.mean([r["switches_last_1000"] for r in group])), 3),
            "median_first_stable": "" if np.isinf(median_stable) else median_stable,
            "mean_wall_clock_s": round(float(np.mean([r["wall_clock_s"] for r in group])), 3),
        })
    return summary


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(out_dir, jobs, n_seeds, episodes_override):
    seeds = sweep_seeds(n_seeds)
    variants = sweep_variants(episodes_override)
    tasks = [(name, method, params, seed) for name, method, params in variants for seed in seeds]
    # More workers than tasks would sit idle, and Windows refuses more than 61 worker processes.
    jobs = min(jobs, len(tasks))
    if sys.platform == "win32":
        jobs = min(jobs, 61)
    print(f"Sweep: {len(variants)} variants x {n_seeds} seeds = {len(tasks)} runs, "
          f"seeds {seeds[0]}..{seeds[-1]}, {jobs} worker(s)", flush=True)
    start_time = time.perf_counter()
    rows = [None] * len(tasks)

    def report(n_done, row):
        print(f"[{n_done:4d}/{len(tasks)}] {row['method']:7s} {row['variant']:15s} "
              f"seed {row['seed']}: optimal={row['is_optimal']} gap={row['gap']:.4f} "
              f"q_rmse={row['q_rmse']:.4f} ({row['wall_clock_s']:.1f} s)", flush=True)

    if jobs <= 1:
        for i, task in enumerate(tasks):
            rows[i] = run_sweep_task(task)
            report(i + 1, rows[i])
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            futures = {pool.submit(run_sweep_task, task): i for i, task in enumerate(tasks)}
            for n_done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                i = futures[future]
                rows[i] = future.result()  # stored by task index, so completion order is irrelevant
                report(n_done, rows[i])

    summary = summarize_sweep(rows)
    write_csv(os.path.join(out_dir, "sweep_runs.csv"), rows)
    write_csv(os.path.join(out_dir, "sweep_summary.csv"), summary)

    print(f"\n{'variant':16s}{'method':9s}{'optimal':>9s}{'mean_gap':>10s}{'max_gap':>9s}"
          f"{'q_rmse':>9s}{'opt_frac':>10s}{'switches':>10s}{'stable':>8s}{'time_s':>8s}")
    for row in summary:
        median = row["median_first_stable"]
        stable = "never" if median == "" else f"{median:.0f}"
        print(f"{row['variant']:16s}{row['method']:9s}"
              f"{str(row['seeds_optimal']) + '/' + str(row['n_seeds']):>9s}"
              f"{row['mean_gap']:10.4f}{row['max_gap']:9.4f}{row['mean_q_rmse']:9.4f}"
              f"{row['mean_optimal_frac']:10.4f}{row['mean_switches']:10.1f}{stable:>8s}"
              f"{row['mean_wall_clock_s']:8.1f}")
    print(f"\nSweep finished in {time.perf_counter() - start_time:.1f} s. "
          f"Wrote sweep_runs.csv and sweep_summary.csv to {out_dir}", flush=True)
    print(f"The times were measured with {jobs} worker(s) running at once: compare them within "
          "this sweep only, not with the main run.", flush=True)


# --------------------------------------------------------------------------
# 12. Figures
# --------------------------------------------------------------------------
# No chart titles (captions live in the report) and never a second y axis. Tabular is
# dashed as well as orange, so the two methods stay apart in a black-and-white print.
COLOR_NEURAL = "#2a78d6"
COLOR_RAW = "#9ec5f4"
COLOR_ROLLING = "#1c5cab"
COLOR_TABULAR = "#eb6834"
COLOR_REFERENCE = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_SPINE = "#c3c2b7"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
EPISODE_WINDOW = 100  # episodes in the rolling means of the cost and of the optimal share
LOSS_WINDOW = 500     # learning steps in the rolling mean of the loss


def apply_figure_style():
    matplotlib.rcParams.update({
        "figure.figsize": (3.45, 2.05),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "axes.edgecolor": COLOR_SPINE,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR_GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "xtick.color": COLOR_SPINE,
        "ytick.color": COLOR_SPINE,
        "xtick.labelcolor": COLOR_TEXT_MUTED,
        "ytick.labelcolor": COLOR_TEXT_MUTED,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "legend.frameon": False,
        "legend.labelcolor": COLOR_TEXT_MUTED,
    })


def finish_figure(fig, ax, out_dir, name):
    # The legend sits above the axes, in the space a title would take, so it never covers data.
    # The labels are kept short so that one row fits even in DejaVu Sans, the wider font that
    # matplotlib falls back to where Arial is missing.
    legend = fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3,
                        borderaxespad=0.1, borderpad=0.1, handlelength=1.5, handletextpad=0.4,
                        columnspacing=0.8, markerscale=4)
    for handle in legend.get_lines():
        handle.set_linewidth(max(handle.get_linewidth(), 1.0))  # a hairline key would be invisible
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
    for extension in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{extension}"), dpi=300)
    plt.close(fig)


def plot_seed_cost(episode_costs, v_star_a, out_dir, name):
    fig, ax = plt.subplots()
    episodes = np.arange(1, len(episode_costs) + 1)
    # Costs are discrete (8, 9, 11), so raw values are small markers: a line would only
    # fill the space between the levels with a solid block.
    ax.plot(episodes, episode_costs, linestyle="none", marker=".", markersize=1.5,
            markeredgewidth=0, color=COLOR_RAW, rasterized=True, label="Episode cost")
    ends, means = rolling_mean(episode_costs, EPISODE_WINDOW)
    # A run no longer than the window gives at most one point: no line is drawn, but the legend
    # would still list it.
    if len(ends) > 1:
        ax.plot(ends, means, color=COLOR_ROLLING, linewidth=1.0,
                label=f"{EPISODE_WINDOW}-episode mean")
    ax.axhline(v_star_a, color=COLOR_REFERENCE, linestyle="--", linewidth=0.8,
               label=f"V*(a) = {v_star_a:.2f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode cost")
    ax.set_yticks([8, 9, 10, 11])
    # The costs lie exactly on 8, 9 and 11, where horizontal gridlines would hide them.
    ax.grid(axis="y", visible=False)
    finish_figure(fig, ax, out_dir, name)


def plot_seed_loss(losses, noise_floor, loss_name, out_dir, name):
    fig, ax = plt.subplots()
    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, color=COLOR_RAW, linewidth=0.25, rasterized=True, label="Raw loss")
    ends, means = rolling_mean(losses, LOSS_WINDOW)
    if len(ends) > 1:  # the same guard as for the episode cost
        ax.plot(ends, means, color=COLOR_ROLLING, linewidth=1.0, label=f"{LOSS_WINDOW}-step mean")
    if noise_floor is not None:  # an MSE quantity, recorded for the MSE loss only
        ax.axhline(noise_floor, color=COLOR_REFERENCE, linestyle="--", linewidth=0.8,
                   label=f"Noise floor = {noise_floor:.3f}")
    ax.set_yscale("log")
    readable_log_axis(ax.yaxis, [0.01, 0.1, 1, 10])  # plain numbers, not tiny exponents
    ax.set_xlabel("Learning step")
    ax.set_ylabel(f"Training loss ({loss_name.upper()})")
    finish_figure(fig, ax, out_dir, name)


def plot_neural_vs_tabular(neural_xy, tabular_xy, ylabel, linewidth):
    fig, ax = plt.subplots()
    ax.plot(*neural_xy, color=COLOR_NEURAL, linewidth=linewidth, label="Neural")
    ax.plot(*tabular_xy, color=COLOR_TABULAR, linewidth=linewidth, linestyle="--",
            label="Tabular (matched schedule)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    return fig, ax


def readable_log_axis(axis, ticks):
    """Label a log axis at plain numbers such as 0.1, 0.2, 0.5, not only at powers of ten."""
    axis.set_major_locator(ticker.FixedLocator(ticks))
    axis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(ticker.NullFormatter())


def build_figures(out_dir):
    """Rebuild every figure from the logs and results.json in out_dir; no training happens here."""
    results_path = os.path.join(out_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"No results.json in {out_dir}: run the experiment first.", flush=True)
        sys.exit(1)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    reference = results["reference"]
    seeds = sorted(int(seed) for seed in results["neural"])
    missing = [name for seed in seeds for name in [f"training_log_seed{seed}.npz",
                                                   f"tabular_log_seed{seed}.npz"]
               if not os.path.exists(os.path.join(out_dir, name))]
    if missing:
        print(f"Missing in {out_dir}: {', '.join(missing)}. Run the experiment again.", flush=True)
        sys.exit(1)

    neural_logs = {}
    tabular_logs = {}
    for seed in seeds:
        with np.load(os.path.join(out_dir, f"training_log_seed{seed}.npz")) as log:
            neural_logs[seed] = {key: log[key] for key in log.files}
        with np.load(os.path.join(out_dir, f"tabular_log_seed{seed}.npz")) as log:
            tabular_logs[seed] = {key: log[key] for key in log.files}

    apply_figure_style()
    written = []
    if 4 in seeds:
        plot_seed_cost(neural_logs[4]["episode_costs"], reference["v_star"]["a"], out_dir,
                       "fig_seed4_cost")
        written.append("fig_seed4_cost")
        losses = neural_logs[4]["losses"]
        if len(losses) > 0:
            plot_seed_loss(losses, results["neural"]["4"]["loss_noise_floor"],
                           results["config"]["loss"], out_dir, "fig_seed4_loss")
            written.append("fig_seed4_loss")
        else:
            print("Seed 4 has no learning steps (run shorter than warm-up): no loss figure.",
                  flush=True)
    else:
        print("Seed 4 was not run: no seed-4 figures.", flush=True)

    neural_metrics = [stability_metrics(neural_logs[s]["q_hist"], reference) for s in seeds]
    tabular_metrics = [stability_metrics(tabular_logs[s]["matched_q_hist"], reference)
                       for s in seeds]
    # A line needs two points, so the runs must be longer than the window of the rolling means.
    if len(neural_metrics[0]["greedy_ok"]) <= EPISODE_WINDOW:
        print(f"The runs are not longer than {EPISODE_WINDOW} episodes, the window of the rolling "
              "means: no stability or Q-error figure.", flush=True)
    else:
        # A log episode axis, because every seed learns within a few hundred episodes, which a
        # linear axis squeezes into its first 5%, and the later dips last only a few episodes.
        # The label names the smoothing: the point at episode k averages episodes k-99 .. k.
        fig, ax = plot_neural_vs_tabular(
            rolling_mean(np.mean([m["greedy_ok"] for m in neural_metrics], axis=0),
                         EPISODE_WINDOW),
            rolling_mean(np.mean([m["greedy_ok"] for m in tabular_metrics], axis=0),
                         EPISODE_WINDOW),
            f"Share of seeds optimal\n({EPISODE_WINDOW}-episode rolling mean)", 1.0)
        ax.set_ylim(-0.03, 1.03)  # a share, so its whole range stays in view
        ax.set_xscale("log")
        n_episodes = len(neural_metrics[0]["greedy_ok"])
        ticks = [tick for tick in [100, 200, 500, 1000, 2000, 5000] if tick <= n_episodes]
        if len(ticks) < 2:  # a short test run: label its last episode too, so the range reads
            ticks.append(n_episodes)
        readable_log_axis(ax.xaxis, ticks)
        finish_figure(fig, ax, out_dir, "fig_stability")
        written.append("fig_stability")

        # The same rolling mean, named in the axis label, because the episode-to-episode jitter of
        # the mean error is as large as the neural-tabular difference, which it would hide. A log
        # axis, because the error falls by two orders of magnitude and then the differences that
        # matter are around 0.1.
        fig, ax = plot_neural_vs_tabular(
            rolling_mean(np.mean([m["q_rmse_curve"] for m in neural_metrics], axis=0),
                         EPISODE_WINDOW),
            rolling_mean(np.mean([m["q_rmse_curve"] for m in tabular_metrics], axis=0),
                         EPISODE_WINDOW),
            f"Mean RMSE of Q against Q*\n({EPISODE_WINDOW}-episode rolling mean)", 1.0)
        ax.set_yscale("log")
        readable_log_axis(ax.yaxis, [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
        finish_figure(fig, ax, out_dir, "fig_q_rmse")
        written.append("fig_q_rmse")
    if written:
        print(f"Wrote figures to {out_dir}: {', '.join(written)} (.pdf and .png)", flush=True)


# --------------------------------------------------------------------------
# 13. Main run
# --------------------------------------------------------------------------
METHODS = ["neural", "tabular_a2", "tabular_matched"]


def run_metadata(cfg, seeds):
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha256 = hashlib.sha256(f.read()).hexdigest()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        # The weights repeat bit for bit only with the same torch build and CPU instruction set,
        # so the instruction set torch uses is recorded, and the exact script that ran.
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "script_sha256": script_sha256,
        "official_run": cfg == NeuralConfig() and list(seeds) == [0, 1, 2, 3, 4],
    }


def run_protocol(cfg):
    """The fixed choices that NeuralConfig does not hold, recorded with the results so that the
    hyperparameter table can be written from the run itself."""
    return {
        "network": " -> ".join([f"one-hot {N_STATES}"] + [f"{width} ReLU" for width in cfg.hidden]
                               + [f"{N_ACTIONS} linear ({', '.join(ACTION_LABELS)})"]),
        "optimizer": "Adam over the online network, torch defaults except lr: betas (0.9, 0.999), "
                     "eps 1e-8, no weight decay",
        "training_streams": "random, numpy and torch seeded with the seed; "
                            "SeedSequence(seed).spawn(3) for environment, exploration and replay",
        "tabular_stream": "default_rng(seed), as in Assignment 2",
        "evaluation": {"episodes": EVAL_EPISODES, "max_steps": EVAL_MAX_STEPS,
                       "noise_stream": f"SeedSequence([{EVAL_SEED_ROOT}, seed]), the same for "
                                       "every method of a seed",
                       "tie_stream": f"SeedSequence([{EVAL_SEED_ROOT}, seed, 1])"},
        "rolling_windows": {"episodes": EPISODE_WINDOW, "learning_steps": LOSS_WINDOW},
    }


def across_seeds(records):
    """Mean and sample standard deviation (ddof 1) across seeds of the main per-seed numbers;
    None where some seed has no value, as for a run that never became stable."""
    def mean_sd(values):
        if any(value is None for value in values):
            return None
        return {"mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}

    summary = {"n_seeds": len(records),
               "seeds_optimal": sum(r["is_optimal"] for r in records),
               "eval_mean": mean_sd([r["eval"]["mean"] for r in records]),
               "eval_std": mean_sd([r["eval"]["std"] for r in records])}
    for key in ["exact_expected_cost", "gap", "q_rmse", "first_stable_episode",
                "switches_last_1000", "rolling_cost_last_100", "wall_clock_s"]:
        summary[key] = mean_sd([r[key] for r in records])
    return summary


def run_seed(seed, cfg, out_dir):
    """Train and evaluate all three methods on one seed, save the logs, return their records."""
    mask = feasibility_mask(PathEnv(np.random.default_rng(0)))
    records = {}

    run = train_neural(seed, cfg)
    counts = (run["n_transitions"], run["n_learning_steps"], run["n_target_syncs"])
    if counts != expected_counts(cfg):
        raise RuntimeError(f"seed {seed}: counts {counts} differ from {expected_counts(cfg)}")
    reference = reference_values()  # for the analysis below; no learner ever receives it
    net = run["online_net"]
    policy = greedy_policy(all_state_q_values(net), mask, evaluation_tie_stream(seed))
    act = neural_actor(net, mask, evaluation_tie_stream(seed))
    record = method_record(seed, policy, act, run, run["n_learning_steps"], reference)
    record["n_target_syncs"] = run["n_target_syncs"]
    record["mean_loss_last_1000"] = (float(np.mean(run["losses"][-1000:]))
                                     if len(run["losses"]) > 0 else None)
    # The floor is an expected squared error, so it is no reference for a Huber loss.
    record["loss_noise_floor"] = (loss_noise_floor(run["replay_counts"]) if cfg.loss == "mse"
                                  else None)
    record["weights_sha256"] = run["weights_sha256"]
    records["neural"] = record
    np.savez_compressed(os.path.join(out_dir, f"training_log_seed{seed}.npz"),
                        episode_costs=run["episode_costs"], epsilons=run["epsilons"],
                        losses=run["losses"], q_hist=run["q_hist"])

    tabular_logs = {}
    for method, prefix in [("tabular_a2", "a2"), ("tabular_matched", "matched")]:
        run = run_q_learning(seed, **tabular_settings(cfg)[method])
        policy = greedy_policy(table_to_matrix(run["Q"]), mask, evaluation_tie_stream(seed))
        records[method] = method_record(seed, policy, table_actor(policy), run,
                                        run["n_transitions"], reference)
        tabular_logs[f"{prefix}_episode_costs"] = run["episode_costs"]
        tabular_logs[f"{prefix}_epsilons"] = run["epsilons"]
        tabular_logs[f"{prefix}_q_hist"] = run["q_hist"]
    np.savez_compressed(os.path.join(out_dir, f"tabular_log_seed{seed}.npz"), **tabular_logs)
    return records


def write_tables(results, out_dir):
    evaluation_rows = []
    policy_rows = []
    for method in METHODS:
        for seed, record in results[method].items():
            stats = record["eval"]
            counts = " ".join(f"{cost}:{n}" for cost, n in stats["cost_counts"].items())
            evaluation_rows.append({
                "method": method,
                "seed": seed,
                "mean_cost": round(stats["mean"], 6),
                "std_cost": round(stats["std"], 6),
                "min_cost": stats["min"],
                "max_cost": stats["max"],
                "pct_reach_h": stats["pct_reach_h"],
                "ci95_half_width": round(stats["ci95_half_width"], 6),
                "cost_counts": counts,
                "exact_expected_cost": round(record["exact_expected_cost"], 6),
                "gap": round(record["gap"], 6),
                "is_optimal": record["is_optimal"],
            })
            row = {"method": method, "seed": seed}
            row.update({f"pi_{s}": record["policy"][s] for s in STATES if s != TERMINAL})
            row.update({
                "nominal_path": record["nominal_path"],
                "nominal_cost": record["nominal_cost"],
                "exact_expected_cost": round(record["exact_expected_cost"], 6),
                "gap": round(record["gap"], 6),
                "is_optimal": record["is_optimal"],
            })
            policy_rows.append(row)
    write_csv(os.path.join(out_dir, "five_seed_evaluation.csv"), evaluation_rows)
    write_csv(os.path.join(out_dir, "policies.csv"), policy_rows)


def print_tables(results):
    reference = results["reference"]
    pi_star = reference["pi_star"]
    print(f"\nReference (policy iteration): V*(a) = {reference['v_star']['a']:.4f}, "
          f"pi* = {pi_star['a']}, {pi_star['b']}, {pi_star['c']}")
    print(f"Network: {results['config']['hidden']} hidden ReLU units, "
          f"{results['parameter_count']} trainable parameters")
    for method in METHODS:
        print(f"\n{method}: greedy policy and {EVAL_EPISODES}-episode evaluation")
        print(f"{'seed':>4s}  {'pi(a,b,c)':10s} {'nominal path':14s}{'cost':>4s} {'mean':>7s} "
              f"{'std':>6s} {'min':>3s} {'max':>3s} {'%h':>6s} {'ci95':>6s} {'exact':>7s} "
              f"{'gap':>6s} {'opt':>3s}")
        for seed, r in results[method].items():
            stats = r["eval"]
            triple = " ".join(r["policy"][s] for s in BRANCHING_STATES)
            optimal = "yes" if r["is_optimal"] else "NO"
            print(f"{seed:>4s}  {triple:10s} {r['nominal_path']:14s}{r['nominal_cost']:4d} "
                  f"{stats['mean']:7.4f} {stats['std']:6.4f} {stats['min']:3.0f} "
                  f"{stats['max']:3.0f} {stats['pct_reach_h']:6.1f} "
                  f"{stats['ci95_half_width']:6.4f} {r['exact_expected_cost']:7.4f} "
                  f"{r['gap']:6.4f} {optimal:>3s}")
        print(f"{'seed':>4s}  {'opt_frac_1k':>11s} {'switches_1k':>11s} {'stable_from':>11s} "
              f"{'q_rmse':>7s} {'roll_100':>8s} {'transitions':>11s} {'updates':>7s} "
              f"{'time_s':>7s}")
        for seed, r in results[method].items():
            stable = r["first_stable_episode"]
            print(f"{seed:>4s}  {r['optimal_frac_last_1000']:11.4f} {r['switches_last_1000']:11d} "
                  f"{'never' if stable is None else str(stable):>11s} {r['q_rmse']:7.4f} "
                  f"{r['rolling_cost_last_100']:8.4f} {r['n_transitions']:11d} "
                  f"{r['n_learning_steps']:7d} {r['wall_clock_s']:7.1f}")
    print("\nneural: training loss against its noise floor")
    print(f"{'seed':>4s}  {'syncs':>5s} {'loss_last_1k':>12s} {'noise_floor':>11s}  weights_sha256")
    for seed, r in results["neural"].items():
        loss = r["mean_loss_last_1000"]
        floor = r["loss_noise_floor"]
        print(f"{seed:>4s}  {r['n_target_syncs']:5d} "
              f"{'none' if loss is None else format(loss, '.4f'):>12s} "
              f"{'none' if floor is None else format(floor, '.4f'):>11s}  "
              f"{r['weights_sha256'][:16]}")
    print("\nAcross seeds: mean +- sample standard deviation of the per-seed numbers")
    print(f"{'method':16s}{'optimal':>7s}  {'eval mean':>17s}  {'exact cost':>17s}  "
          f"{'q_rmse':>17s}  {'roll_100':>17s}")
    for method in METHODS:
        summary = results["across_seeds"][method]
        cells = []
        for key in ["eval_mean", "exact_expected_cost", "q_rmse", "rolling_cost_last_100"]:
            sd = summary[key]["sd"]  # None for a single seed
            cells.append(f"{summary[key]['mean']:.4f} +- "
                         f"{'-' if sd is None else format(sd, '.4f')}")
        print(f"{method:16s}{str(summary['seeds_optimal']) + '/' + str(summary['n_seeds']):>7s}  "
              + "  ".join(f"{cell:>17s}" for cell in cells))


def remove_old_outputs(out_dir):
    """Delete the logs, tables and figures that an earlier main run left in out_dir, so that no
    file of another run can sit next to the new results. checks.txt and sweep files stay."""
    for name in os.listdir(out_dir):
        if (re.fullmatch(r"(training|tabular)_log_seed\d+\.npz", name)
                or re.fullmatch(r"fig_(seed4_cost|seed4_loss|stability|q_rmse)\.(pdf|png)", name)
                or name in ["results.json", "five_seed_evaluation.csv", "policies.csv"]):
            os.remove(os.path.join(out_dir, name))


def run_main_experiment(seeds, cfg, out_dir):
    start_time = time.perf_counter()
    remove_old_outputs(out_dir)
    records_by_method = {method: {} for method in METHODS}
    for seed in seeds:
        print(f"Seed {seed}: training and evaluating all methods ...", flush=True)
        records = run_seed(seed, cfg, out_dir)
        for method in METHODS:
            records_by_method[method][str(seed)] = records[method]
        neural = records["neural"]
        print(f"Seed {seed}: neural {' '.join(neural['policy'][s] for s in BRANCHING_STATES)}, "
              f"gap {neural['gap']:.4f}, {neural['wall_clock_s']:.1f} s", flush=True)

    results = {
        "meta": run_metadata(cfg, seeds),  # after training, which sets the thread count
        "config": dataclasses.asdict(cfg),
        "protocol": run_protocol(cfg),
        "tabular_settings": tabular_settings(cfg),
        "parameter_count": count_parameters(QNetwork(cfg.hidden)),
        "reference": reference_values(),
        **records_by_method,
        "across_seeds": {method: across_seeds(list(records_by_method[method].values()))
                         for method in METHODS},
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_tables(results, out_dir)
    build_figures(out_dir)
    print_tables(results)
    print(f"\nFinished in {time.perf_counter() - start_time:.1f} s. Results are in {out_dir}",
          flush=True)
    if not results["meta"]["official_run"]:
        print("WARNING: these results are NOT the official ones, which use the default "
              "configuration on seeds 0-4.", flush=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural vs. tabular Q-learning on the episodic shortest-path MDP.")
    mode = parser.add_mutually_exclusive_group()  # at most one of these three modes per call
    mode.add_argument("--checks-only", action="store_true", help="run the self-checks and stop")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="seeds of the main run (default: 0 1 2 3 4)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="training length; anything but the default 5000 is not an official "
                             "result; with --sweep it sets every variant's length (smoke tests)")
    parser.add_argument("--out", default=RESULTS_DIR,
                        help="output folder (default: results/ next to this file)")
    parser.add_argument("--skip-checks", action="store_true",
                        help="do not run the self-checks first")
    mode.add_argument("--plots-only", action="store_true",
                      help="rebuild the figures from the saved logs without training")
    mode.add_argument("--sweep", action="store_true",
                      help="run the sensitivity study instead of the main run")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="worker processes of the sweep (default: half of the CPU count)")
    parser.add_argument("--sweep-seeds", type=int, default=10,
                        help="number of held-out seeds of the sweep (default: 10)")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.sweep_seeds < 1:
        parser.error("--sweep-seeds must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if not all(0 <= seed < 2 ** 32 for seed in args.seeds):
        parser.error("--seeds must be between 0 and 2**32 - 1")
    return args


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)

    if args.plots_only:
        build_figures(out_dir)
        return
    os.makedirs(out_dir, exist_ok=True)
    if args.checks_only:
        run_checks(out_dir)
        return
    if not args.skip_checks:
        run_checks(out_dir)  # exits non-zero, so nothing below runs, if a check fails

    if args.sweep:
        if args.episodes is not None:
            print(f"WARNING: --episodes {args.episodes} overrides every variant's length: "
                  "a smoke test, not the sensitivity study.", flush=True)
        run_sweep(out_dir, args.jobs, args.sweep_seeds, args.episodes)
        return

    cfg = NeuralConfig()
    if args.episodes is not None and args.episodes != cfg.episodes:
        print(f"WARNING: --episodes {args.episodes} is not the default {cfg.episodes}: these "
              "results are NOT the official ones.", flush=True)
        cfg = dataclasses.replace(cfg, episodes=args.episodes,
                                  eps_decay_episodes=round(0.8 * args.episodes))
    run_main_experiment(list(dict.fromkeys(args.seeds)), cfg, out_dir)


if __name__ == "__main__":
    main()
