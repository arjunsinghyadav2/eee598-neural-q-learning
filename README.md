# EEE598 Assignment 3: Neural Q-Learning

**Team 9:** Chinmay Prashanth, Arjun Singh Yadav, Muhammad Ibraheem

A feedforward network replaces the Q table of our Assignment 2 learner on the same episodic
stochastic shortest-path problem (states a to h, start a, terminal h, gamma = 1). It is trained
only from sampled transitions, with experience replay, a hard-copied target network, Adam and a
mean-squared temporal-difference loss, and is compared with tabular Q-learning. The IEEE-format
report is [`report/report.pdf`](report/report.pdf).

## Software and hardware

- Python 3.14.0, PyTorch 2.10.0+cpu, NumPy 2.4.2, Matplotlib 3.10.8
- Intel Core i9-14900KF, 64 GB RAM, Windows 11; CPU only, one thread per training run
- The reported times come from a run pinned to the performance cores (logical CPUs 0-15).
  Under Windows' default scheduling the run can land on the slower efficiency cores and take
  longer; the learned weights and every other result are identical either way.

## How to run

```
pip install -r requirements.txt
python neural_q_learning.py
```

This runs 149 self-checks and stops if any fails. It then trains the network and both tabular
baselines on seeds 0-4 (5000 episodes each), evaluates every greedy policy on 1000 episodes,
and writes the figures and tables to `results/` next to the script. It takes about 1.5 minutes.
To pin it to the performance cores as for the reported times, run in PowerShell:

```
$p = Start-Process python neural_q_learning.py -NoNewWindow -PassThru; $p.ProcessorAffinity = 0xFFFF; $p.WaitForExit()
```

Other modes:

```
python neural_q_learning.py --sweep         # one-factor sensitivity study on held-out seeds 100-109 (about 5 min with 16 workers)
python neural_q_learning.py --plots-only    # redraw the figures from the saved logs
python neural_q_learning.py --checks-only   # self-checks only
```

## Results on seeds 0-4

| | Neural | Tabular (Assignment 2) | Tabular (matched schedule) |
|---|---|---|---|
| Greedy policy at a, b, c | Ab, Ae, Af on 5/5 seeds | same | same |
| Exact expected cost | 8.44 = V*(a) | 8.44 | 8.44 |
| 1000-episode evaluation, mean over seeds | 8.435 ± 0.018 | identical | identical |
| RMSE of Q against Q* | 0.091 ± 0.010 | 0.084 ± 0.023 | 0.097 ± 0.021 |
| Training time per seed | 15.1 s | 0.20 s | 0.18 s |

All methods of a seed are evaluated on the same random stream (common random numbers), so
methods that learned the same policy have identical evaluation statistics.

## Files

- `neural_q_learning.py`: the single source file (environment reused from Assignment 2,
  network, training, tabular baselines, evaluation, sweep, figures and self-checks)
- `requirements.txt`: pinned library versions
- `results/`: `checks.txt`, `results.json`, `five_seed_evaluation.csv`, `policies.csv`,
  `sweep_runs.csv`, `sweep_summary.csv`, per-seed logs (`.npz`) and figures (`.pdf`, `.png`)
- `report/`: `report.tex` and `report.pdf`

Rerunning reproduces the final-weight SHA-256 hashes in `results.json` bit for bit on the same
machine and library build. Other CPUs or builds may differ in the last floating-point digits.

## Disclosure of AI assistance

We acknowledge the use of generative AI: Claude Code (Anthropic) wrote the implementation and
its self-checks, ran the experiments and drafted the report at the team's direction. The team
tested the code and verified the reported results.
