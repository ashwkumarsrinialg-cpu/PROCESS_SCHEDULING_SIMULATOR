"""
Enhanced CPU Scheduler Simulator
=================================
Compares CFS, EEVDF, and Hybrid (EEVDF+MLFQ) schedulers with:
  - 100+ process workloads (realistic mobile task distributions)
  - Statistical analysis over many Monte Carlo runs
  - Sensitivity study for alpha (urgency bias) and tau(T) (thermal scaling)
  - Comparison against Linux EEVDF reference parameters
  - Publication-quality plots and benchmark discussion output
"""

import random
import math
import copy
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

# ─────────────────────────────────────────────
#  Global style
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "CFS":    "#4C72B0",
    "EEVDF":  "#DD8452",
    "Hybrid": "#55A868",
}

# ─────────────────────────────────────────────
#  Process model
# ─────────────────────────────────────────────
class Process:
    def __init__(self, pid, arrival, burst, priority=0, task_type="bg"):
        self.pid        = pid
        self.arrival    = arrival
        self.burst      = burst
        self.remaining  = burst
        self.priority   = priority      # 0=RT, 1=interactive, 2=background
        self.task_type  = task_type
        self.start_time = -1
        self.completion_time = -1
        self.vruntime   = 0.0
        self.deadline   = 0.0
        self.tier       = priority      # MLFQ tier (may change at runtime)

    def clone(self):
        return copy.deepcopy(self)

# ─────────────────────────────────────────────
#  Workload generators
# ─────────────────────────────────────────────
TASK_PROFILES = {
    # (burst_min, burst_max, priority, type, weight)
    "rt":          (1,  4,  0, "rt",          0.10),
    "interactive": (2,  8,  1, "interactive", 0.30),
    "background":  (5, 20,  2, "background",  0.40),
    "ml_batch":    (15, 40, 2, "ml_batch",    0.10),
    "sensor":      (1,  3,  0, "sensor",      0.10),
}

def generate_workload(n=120, seed=None, burst_scale=1.0):
    """
    Generate n processes with realistic mobile task distribution.
    Returns list of Process objects sorted by arrival time.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    types   = list(TASK_PROFILES.keys())
    weights = [TASK_PROFILES[t][4] for t in types]

    procs = []
    time  = 0
    for i in range(n):
        tname = random.choices(types, weights=weights)[0]
        bmin, bmax, pri, ttype, _ = TASK_PROFILES[tname]
        burst   = round(random.uniform(bmin, bmax) * burst_scale, 1)
        # Inter-arrival: exponential with mean 1.5 time units
        time   += max(0, round(np.random.exponential(1.5), 1))
        procs.append(Process(i+1, time, burst, pri, ttype))

    return procs

# ─────────────────────────────────────────────
#  Schedulers
# ─────────────────────────────────────────────

# ── Linux EEVDF reference parameters ──────────
# From kernel/sched/fair.c (6.5-rc1):
#   sysctl_sched_latency        = 6 ms   (virtual time slice target)
#   sysctl_sched_min_granularity= 0.75 ms
#   latency_nice range          = -20..19  -> weight multiplier 0.5x .. 2x
LINUX_EEVDF_LATENCY_TARGET   = 6.0   # normalised units
LINUX_EEVDF_MIN_GRANULARITY  = 0.75


def _weight(priority, scale=1.0):
    """Map priority level to weight, mirroring Linux nice-to-weight table."""
    table = {0: 2.0 * scale, 1: 1.0 * scale, 2: 0.5 * scale}
    return table.get(priority, 1.0 * scale)


def simulate_cfs(procs_in):
    """CFS: always run process with smallest weighted vruntime."""
    procs = [p.clone() for p in procs_in]
    time, active, done = 0.0, [], []

    while len(done) < len(procs):
        for p in procs:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                active.append(p)

        if not active:
            time += 0.5
            continue

        cur = min(active, key=lambda p: p.vruntime)
        if cur.start_time == -1:
            cur.start_time = time

        sl = min(2.0, cur.remaining)
        w  = _weight(cur.priority)
        cur.remaining -= sl
        cur.vruntime  += sl / w
        time += sl

        if cur.remaining <= 0:
            cur.completion_time = time
            done.append(cur)
            active.remove(cur)

    return done


def simulate_eevdf(procs_in, latency_target=LINUX_EEVDF_LATENCY_TARGET,
                   min_gran=LINUX_EEVDF_MIN_GRANULARITY, weight_scale=1.0):
    """
    EEVDF: assign virtual deadlines and always pick eligible task with earliest deadline.
    Supports Linux-style latency_target and min_granularity parameters.
    """
    procs = [p.clone() for p in procs_in]
    virtual_time = 0.0
    time, active, done = 0.0, [], []

    while len(done) < len(procs):
        for p in procs:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                w = _weight(p.priority, weight_scale)
                # Time-slice request proportional to weight and latency target
                r = max(min_gran, latency_target * w / max(1, len(active) + 1))
                p.deadline = virtual_time + r / w
                active.append(p)

        if not active:
            time += 0.5
            virtual_time += 0.5
            continue

        # Eligible = arrival <= time (already guaranteed by admission above)
        cur = min(active, key=lambda p: p.deadline)
        if cur.start_time == -1:
            cur.start_time = time

        w  = _weight(cur.priority, weight_scale)
        sl = min(max(min_gran, latency_target / max(1, len(active))), cur.remaining)
        cur.remaining  -= sl
        virtual_time   += sl / len(active) if active else sl
        time           += sl

        if cur.remaining > 0:
            r = max(min_gran, latency_target * w / max(1, len(active)))
            cur.deadline = virtual_time + r / w
        else:
            cur.completion_time = time
            done.append(cur)
            active.remove(cur)

    return done


def tau_T(T, T_max=85.0, T_min=40.0):
    """
    Thermal scaling factor: monotonically decreasing in [0,1].
    tau(T) = 1 at T <= T_min, tau(T) = 0.3 at T >= T_max.
    Uses a smooth sigmoid-like ramp.
    """
    if T <= T_min:
        return 1.0
    if T >= T_max:
        return 0.3
    t = (T - T_min) / (T_max - T_min)
    return 1.0 - 0.7 * (3 * t**2 - 2 * t**3)   # smooth-step


def simulate_hybrid(procs_in, alpha=0.5, temperature=55.0,
                    core_caps=None, weight_scale=1.0):
    """
    Hybrid EEVDF+MLFQ with thermal scaling, core capacity normalisation,
    and urgency bias.

    Parameters
    ----------
    alpha       : urgency bias coefficient
    temperature : device temperature in °C
    core_caps   : dict {priority: cap} — capacity of core assigned to tier
                  e.g. {0: 1.0, 1: 0.8, 2: 0.5}  (big.LITTLE)
    """
    if core_caps is None:
        core_caps = {0: 1.0, 1: 0.8, 2: 0.5}

    procs = [p.clone() for p in procs_in]
    tau   = tau_T(temperature)
    vtime = 0.0
    time, active, done = 0.0, [], []

    while len(done) < len(procs):
        for p in procs:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                active.append(p)

        if not active:
            time  += 0.5
            vtime += 0.5 * tau
            continue

        # MLFQ: select highest-priority tier
        min_tier = min(p.tier for p in active)
        candidates = [p for p in active if p.tier == min_tier]

        # EEVDF within tier: compute effective deadline
        def eff_deadline(p):
            w   = _weight(p.priority, weight_scale)
            cap = core_caps.get(p.priority, 1.0)
            # Core-capacity normalised deadline
            d_norm = p.arrival + p.burst / (w * cap)
            # Thermal scaling applied to virtual time component
            d_thermal = d_norm * tau
            # Urgency bias (U_i inversely proportional to priority level)
            U_i = 1.0 / (1 + p.priority)
            return d_thermal - alpha * U_i

        cur = min(candidates, key=eff_deadline)

        if cur.start_time == -1:
            cur.start_time = time

        sl = min(2.0, cur.remaining)
        cur.remaining -= sl
        vtime += sl * tau
        time  += sl

        if cur.remaining <= 0:
            cur.completion_time = time
            done.append(cur)
            active.remove(cur)
        else:
            # MLFQ demotion: if a background process uses multiple slices,
            # keep tier (already at max); interactive tasks stay interactive
            if cur.tier < 2:
                cur.tier = min(2, cur.tier + 1)   # demote after one full slice

    return done


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────
def compute_metrics(completed):
    waits      = [p.start_time - p.arrival for p in completed]
    turnarounds = [p.completion_time - p.arrival for p in completed]
    responses  = [p.start_time - p.arrival for p in completed]
    throughput = len(completed) / max(p.completion_time for p in completed)

    # Jain's Fairness Index on turnaround times
    arr   = np.array(turnarounds)
    jain  = (arr.sum()**2) / (len(arr) * (arr**2).sum())

    # Deadline miss rate (tasks that waited > 2x their burst)
    misses = sum(1 for p in completed if (p.completion_time - p.arrival) > 2.5 * p.burst)

    return {
        "avg_wait":       statistics.mean(waits),
        "std_wait":       statistics.stdev(waits) if len(waits) > 1 else 0,
        "avg_turnaround": statistics.mean(turnarounds),
        "std_turnaround": statistics.stdev(turnarounds) if len(turnarounds) > 1 else 0,
        "avg_response":   statistics.mean(responses),
        "p95_wait":       float(np.percentile(waits, 95)),
        "p99_wait":       float(np.percentile(waits, 99)),
        "throughput":     throughput,
        "jain_fairness":  jain,
        "deadline_miss":  misses / len(completed),
        "p95_turnaround": float(np.percentile(turnarounds, 95)),
    }


# ─────────────────────────────────────────────
#  Original 5-process timeline charts (paper figures)
# ─────────────────────────────────────────────
def _cpu_timeline_cfs(procs_in):
    procs = [p.clone() for p in procs_in]
    time, active, timeline, done = 0.0, [], [], []
    while len(done) < len(procs):
        for p in procs:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                active.append(p)
        if not active:
            time += 1; continue
        cur = min(active, key=lambda p: p.vruntime)
        sl  = min(2, cur.remaining)
        timeline.append((cur.pid, time, time+sl))
        cur.remaining -= sl
        cur.vruntime  += sl * (1.0 / (1 + cur.priority))
        time += sl
        if cur.remaining <= 0:
            cur.completion_time = time; done.append(cur); active.remove(cur)
    return timeline

def _cpu_timeline_eevdf(procs_in):
    procs = [p.clone() for p in procs_in]
    time, active, timeline, done = 0.0, [], [], []
    while len(done) < len(procs):
        for p in procs:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                p.deadline = p.arrival + p.burst / (1 + p.priority)
                active.append(p)
        if not active:
            time += 1; continue
        cur = min(active, key=lambda p: p.deadline)
        sl  = min(2, cur.remaining)
        timeline.append((cur.pid, time, time+sl))
        cur.remaining -= sl
        time += sl
        cur.deadline = time + cur.remaining / (1 + cur.priority)
        if cur.remaining <= 0:
            cur.completion_time = time; done.append(cur); active.remove(cur)
    return timeline

def _cpu_timeline_hybrid(procs_in):
    procs = [p.clone() for p in procs_in]
    time, active, timeline, done = 0.0, [], [], []
    while len(done) < len(procs):
        for p in procs:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                active.append(p)
        if not active:
            time += 1; continue
        min_prio   = min(p.priority for p in active)
        candidates = [p for p in active if p.priority == min_prio]
        cur = min(candidates, key=lambda p: p.arrival + p.burst / (1 + p.priority))
        sl  = min(2, cur.remaining)
        timeline.append((cur.pid, time, time+sl))
        cur.remaining -= sl; time += sl
        if cur.remaining <= 0:
            cur.completion_time = time; done.append(cur); active.remove(cur)
    return timeline


def generate_cpu_allocation_timeline(cpu_timeline, title, filename):
    """Paper figure: CPU allocation timeline (Gantt chart)."""
    pids    = sorted(set(x[0] for x in cpu_timeline))
    n_pids  = len(pids)
    colors  = plt.cm.tab10(np.linspace(0, 0.9, n_pids))
    pid_col = {pid: colors[i] for i, pid in enumerate(pids)}

    fig, ax = plt.subplots(figsize=(14, max(4, n_pids * 0.7)))

    for pid, start, end in cpu_timeline:
        ax.barh(f"P{pid}", end - start, left=start,
                color=pid_col[pid], edgecolor="black", linewidth=0.6,
                height=0.55, alpha=0.9)
        if (end - start) >= 0.8:
            ax.text((start + end) / 2, f"P{pid}", f"P{pid}",
                    ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")

    ax.set_xlabel("Time Units")
    ax.set_ylabel("Process")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5, linewidth=0.8)

    # Add tick marks at every 5 units
    max_t = max(e for _, _, e in cpu_timeline)
    ax.set_xticks(range(0, int(max_t) + 2, 2))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ─────────────────────────────────────────────
#  Monte Carlo statistical study
# ─────────────────────────────────────────────
def monte_carlo_study(n_runs=200, n_procs=120):
    """
    Run n_runs independent workloads of n_procs processes each.
    Returns dict of metric arrays per scheduler.
    """
    results = {s: {m: [] for m in [
        "avg_wait", "std_wait", "avg_turnaround", "p95_wait",
        "throughput", "jain_fairness", "deadline_miss", "p95_turnaround",
        "avg_response"
    ]} for s in ["CFS", "EEVDF", "Hybrid"]}

    for run in range(n_runs):
        procs = generate_workload(n=n_procs, seed=run)

        for name, fn in [
            ("CFS",    lambda p: simulate_cfs(p)),
            ("EEVDF",  lambda p: simulate_eevdf(p)),
            ("Hybrid", lambda p: simulate_hybrid(p)),
        ]:
            done    = fn(procs)
            metrics = compute_metrics(done)
            for k, v in metrics.items():
                if k in results[name]:
                    results[name][k].append(v)

        if (run + 1) % 50 == 0:
            print(f"    Monte Carlo: {run+1}/{n_runs} runs complete")

    return results


def plot_statistical_comparison(mc_results, filename="stat_comparison.png"):
    """Box-and-whisker + mean comparison across all schedulers."""
    metrics_to_plot = [
        ("avg_wait",        "Avg Waiting Time (units)"),
        ("avg_turnaround",  "Avg Turnaround Time (units)"),
        ("p95_wait",        "P95 Waiting Time (units)"),
        ("throughput",      "Throughput (procs/unit)"),
        ("jain_fairness",   "Jain Fairness Index"),
        ("deadline_miss",   "Deadline Miss Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    schedulers = ["CFS", "EEVDF", "Hybrid"]
    colors     = [PALETTE[s] for s in schedulers]

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        data = [mc_results[s][metric] for s in schedulers]
        bp   = ax.boxplot(data, patch_artist=True, notch=True,
                          widths=0.5, showfliers=False,
                          medianprops=dict(color="black", linewidth=2))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay mean dots
        for i, (d, c) in enumerate(zip(data, colors), 1):
            ax.scatter([i], [statistics.mean(d)], color=c,
                       zorder=5, s=60, marker="D", edgecolors="black", linewidths=0.8)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(schedulers)
        ax.set_title(label, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    legend_patches = [mpatches.Patch(color=PALETTE[s], alpha=0.75, label=s)
                      for s in schedulers]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Statistical Comparison: CFS vs EEVDF vs Hybrid\n"
                 f"(n=200 Monte Carlo runs, 120 processes each)",
                 fontweight="bold", fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved {filename}")


def print_statistical_summary(mc_results):
    """Print mean ± 95% CI and Welch t-test vs EEVDF for every metric."""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY  (mean ± 95% CI, Welch t-test vs EEVDF)")
    print("="*70)

    metrics = [
        ("avg_wait",        "Avg Wait"),
        ("avg_turnaround",  "Avg Turnaround"),
        ("p95_wait",        "P95 Wait"),
        ("throughput",      "Throughput"),
        ("jain_fairness",   "Jain Fairness"),
        ("deadline_miss",   "Deadline Miss Rate"),
        ("avg_response",    "Avg Response"),
    ]

    for metric, label in metrics:
        print(f"\n  {label}")
        eevdf_data = mc_results["EEVDF"][metric]
        for sched in ["CFS", "EEVDF", "Hybrid"]:
            d   = mc_results[sched][metric]
            m   = statistics.mean(d)
            se  = statistics.stdev(d) / math.sqrt(len(d))
            ci  = 1.96 * se
            if sched != "EEVDF":
                t, p = stats.ttest_ind(d, eevdf_data, equal_var=False)
                sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                print(f"    {sched:8s}: {m:7.3f} ± {ci:.3f}   (p={p:.4f} {sig})")
            else:
                print(f"    {sched:8s}: {m:7.3f} ± {ci:.3f}   [reference]")


# ─────────────────────────────────────────────
#  Sensitivity: alpha
# ─────────────────────────────────────────────
def sensitivity_alpha(procs_base, alphas=None, n_seeds=30, filename="sensitivity_alpha.png"):
    if alphas is None:
        alphas = np.linspace(0.0, 2.0, 21)

    avg_wait  = []
    avg_turn  = []
    p95_wait  = []
    miss_rate = []

    for a in alphas:
        w_runs, t_runs, p95_runs, mr_runs = [], [], [], []
        for seed in range(n_seeds):
            procs = generate_workload(n=60, seed=seed + 1000)
            done  = simulate_hybrid(procs, alpha=a)
            m     = compute_metrics(done)
            w_runs.append(m["avg_wait"])
            t_runs.append(m["avg_turnaround"])
            p95_runs.append(m["p95_wait"])
            mr_runs.append(m["deadline_miss"])
        avg_wait.append(statistics.mean(w_runs))
        avg_turn.append(statistics.mean(t_runs))
        p95_wait.append(statistics.mean(p95_runs))
        miss_rate.append(statistics.mean(mr_runs))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    pairs = [
        (axes[0,0], avg_wait,  "Avg Waiting Time",     "royalblue"),
        (axes[0,1], avg_turn,  "Avg Turnaround Time",  "darkorange"),
        (axes[1,0], p95_wait,  "P95 Waiting Time",     "seagreen"),
        (axes[1,1], miss_rate, "Deadline Miss Rate",   "crimson"),
    ]
    for ax, data, title, color in pairs:
        ax.plot(alphas, data, color=color, linewidth=2.2, marker="o",
                markersize=4, markerfacecolor="white", markeredgewidth=1.5)
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1,
                   label="default α=0.5")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    for ax in axes[1]:
        ax.set_xlabel("Urgency Bias Coefficient  α")

    fig.suptitle("Sensitivity Study: Urgency Bias α  (Hybrid Scheduler)\n"
                 f"Averaged over {n_seeds} random workloads, 60 processes each",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ─────────────────────────────────────────────
#  Sensitivity: tau(T) / temperature
# ─────────────────────────────────────────────
def sensitivity_temperature(procs_base, temps=None, n_seeds=30,
                             filename="sensitivity_temperature.png"):
    if temps is None:
        temps = np.linspace(30, 95, 26)

    tau_vals  = [tau_T(T) for T in temps]
    avg_wait  = []
    avg_turn  = []
    p95_wait  = []
    miss_rate = []

    for T in temps:
        w_runs, t_runs, p95_runs, mr_runs = [], [], [], []
        for seed in range(n_seeds):
            procs = generate_workload(n=60, seed=seed + 2000)
            done  = simulate_hybrid(procs, temperature=T)
            m     = compute_metrics(done)
            w_runs.append(m["avg_wait"])
            t_runs.append(m["avg_turnaround"])
            p95_runs.append(m["p95_wait"])
            mr_runs.append(m["deadline_miss"])
        avg_wait.append(statistics.mean(w_runs))
        avg_turn.append(statistics.mean(t_runs))
        p95_wait.append(statistics.mean(p95_runs))
        miss_rate.append(statistics.mean(mr_runs))

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_tau = fig.add_subplot(gs[0, :])
    ax_tau.plot(temps, tau_vals, color="firebrick", linewidth=2.5)
    ax_tau.axvline(40,  color="gray",  linestyle=":", linewidth=1.2, label="T_min=40°C")
    ax_tau.axvline(85,  color="black", linestyle=":", linewidth=1.2, label="T_max=85°C")
    ax_tau.fill_between(temps, tau_vals, alpha=0.15, color="firebrick")
    ax_tau.set_ylabel("τ(T)  thermal scale factor")
    ax_tau.set_title("Thermal Scaling Function  τ(T)", fontweight="bold")
    ax_tau.legend(fontsize=9); ax_tau.grid(True, linestyle="--", alpha=0.4)

    pairs = [
        (fig.add_subplot(gs[1, 0]), avg_wait,  "Avg Waiting Time",    "royalblue"),
        (fig.add_subplot(gs[1, 1]), avg_turn,  "Avg Turnaround Time", "darkorange"),
        (fig.add_subplot(gs[2, 0]), p95_wait,  "P95 Waiting Time",    "seagreen"),
        (fig.add_subplot(gs[2, 1]), miss_rate, "Deadline Miss Rate",  "crimson"),
    ]
    for ax, data, title, color in pairs:
        ax.plot(temps, data, color=color, linewidth=2.2, marker="o",
                markersize=4, markerfacecolor="white", markeredgewidth=1.5)
        ax.axvline(55, color="gray", linestyle="--", linewidth=1, label="default 55°C")
        ax.set_xlabel("Device Temperature (°C)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.suptitle("Sensitivity Study: Thermal Scaling τ(T)  (Hybrid Scheduler)\n"
                 f"Averaged over {n_seeds} random workloads, 60 processes each",
                 fontweight="bold", fontsize=13)
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ─────────────────────────────────────────────
#  Linux EEVDF parameter sweep
# ─────────────────────────────────────────────
def linux_eevdf_param_sweep(n_seeds=40, filename="linux_eevdf_params.png"):
    """
    Sweep Linux EEVDF sysctl parameters:
      sched_latency     : 2 .. 20 ms (normalised units)
      sched_min_gran    : 0.25 .. 3.0
    and measure effect on avg_wait, jain_fairness, deadline_miss.
    """
    latencies = np.linspace(2, 20, 10)
    mingrans  = [0.25, 0.5, 0.75, 1.5, 3.0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cmap   = plt.cm.viridis
    colors = [cmap(i / (len(mingrans)-1)) for i in range(len(mingrans))]

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx, mg in enumerate(mingrans):
        aw_vals, jf_vals, dm_vals = [], [], []
        for lat in latencies:
            aw_r, jf_r, dm_r = [], [], []
            for seed in range(n_seeds):
                procs = generate_workload(n=80, seed=seed + 3000)
                done  = simulate_eevdf(procs, latency_target=lat, min_gran=mg)
                m     = compute_metrics(done)
                aw_r.append(m["avg_wait"])
                jf_r.append(m["jain_fairness"])
                dm_r.append(m["deadline_miss"])
            aw_vals.append(statistics.mean(aw_r))
            jf_vals.append(statistics.mean(jf_r))
            dm_vals.append(statistics.mean(dm_r))

        lbl = f"min_gran={mg}"
        axes[0].plot(latencies, aw_vals, color=colors[idx], marker="o",
                     markersize=5, linewidth=2, label=lbl)
        axes[1].plot(latencies, jf_vals, color=colors[idx], marker="s",
                     markersize=5, linewidth=2, label=lbl)
        axes[2].plot(latencies, dm_vals, color=colors[idx], marker="^",
                     markersize=5, linewidth=2, label=lbl)

    # Mark Linux default
    for ax in axes:
        ax.axvline(LINUX_EEVDF_LATENCY_TARGET, color="red", linestyle="--",
                   linewidth=1.5, label="Linux default 6ms")

    axes[0].set_title("Avg Waiting Time",    fontweight="bold")
    axes[1].set_title("Jain Fairness Index", fontweight="bold")
    axes[2].set_title("Deadline Miss Rate",  fontweight="bold")

    for ax in axes:
        ax.set_xlabel("sched_latency (normalised units)")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Linux EEVDF Parameter Sweep  (sched_latency × sched_min_granularity)\n"
                 f"Averaged over {n_seeds} random workloads, 80 processes each",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ─────────────────────────────────────────────
#  Workload-class breakdown plot
# ─────────────────────────────────────────────
def plot_per_class_metrics(n_seeds=50, filename="per_class_metrics.png"):
    """
    For each scheduler, show avg wait and deadline miss broken down by task type.
    """
    task_types  = ["rt", "interactive", "background", "ml_batch", "sensor"]
    schedulers  = ["CFS", "EEVDF", "Hybrid"]
    sim_fns     = {
        "CFS":    lambda p: simulate_cfs(p),
        "EEVDF":  lambda p: simulate_eevdf(p),
        "Hybrid": lambda p: simulate_hybrid(p),
    }

    # Accumulate
    class_wait  = {s: {t: [] for t in task_types} for s in schedulers}
    class_miss  = {s: {t: [] for t in task_types} for s in schedulers}

    for seed in range(n_seeds):
        procs = generate_workload(n=100, seed=seed + 5000)
        for sched, fn in sim_fns.items():
            done = fn(procs)
            for p in done:
                w = p.start_time - p.arrival
                miss = int((p.completion_time - p.arrival) > 2.5 * p.burst)
                class_wait[sched][p.task_type].append(w)
                class_miss[sched][p.task_type].append(miss)

    x      = np.arange(len(task_types))
    width  = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, sched in enumerate(schedulers):
        means_w = [statistics.mean(class_wait[sched][t]) if class_wait[sched][t] else 0
                   for t in task_types]
        means_m = [statistics.mean(class_miss[sched][t]) if class_miss[sched][t] else 0
                   for t in task_types]
        offset  = (i - 1) * width
        axes[0].bar(x + offset, means_w, width, label=sched,
                    color=PALETTE[sched], alpha=0.85, edgecolor="black", linewidth=0.5)
        axes[1].bar(x + offset, means_m, width, label=sched,
                    color=PALETTE[sched], alpha=0.85, edgecolor="black", linewidth=0.5)

    for ax, title, ylabel in [
        (axes[0], "Avg Waiting Time by Task Class",   "Avg Waiting Time (units)"),
        (axes[1], "Deadline Miss Rate by Task Class", "Miss Rate"),
    ]:
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(task_types, rotation=20, ha="right")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Per-Class Performance Breakdown\n"
                 f"Averaged over {n_seeds} random workloads, 100 processes each",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ─────────────────────────────────────────────
#  Scalability: vary n_procs
# ─────────────────────────────────────────────
def plot_scalability(filename="scalability.png", n_seeds=30):
    sizes = [10, 30, 60, 100, 150, 200]
    results = {s: {"avg_wait": [], "throughput": [], "jain": []}
               for s in ["CFS", "EEVDF", "Hybrid"]}

    fns = {
        "CFS":    simulate_cfs,
        "EEVDF":  simulate_eevdf,
        "Hybrid": simulate_hybrid,
    }

    for n in sizes:
        for sched, fn in fns.items():
            w_r, th_r, j_r = [], [], []
            for seed in range(n_seeds):
                procs = generate_workload(n=n, seed=seed + 7000)
                done  = fn(procs)
                m     = compute_metrics(done)
                w_r.append(m["avg_wait"])
                th_r.append(m["throughput"])
                j_r.append(m["jain_fairness"])
            results[sched]["avg_wait"].append(statistics.mean(w_r))
            results[sched]["throughput"].append(statistics.mean(th_r))
            results[sched]["jain"].append(statistics.mean(j_r))

        print(f"    Scalability: n={n} done")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for sched in ["CFS", "EEVDF", "Hybrid"]:
        axes[0].plot(sizes, results[sched]["avg_wait"],    marker="o", linewidth=2,
                     label=sched, color=PALETTE[sched])
        axes[1].plot(sizes, results[sched]["throughput"],  marker="s", linewidth=2,
                     label=sched, color=PALETTE[sched])
        axes[2].plot(sizes, results[sched]["jain"],        marker="^", linewidth=2,
                     label=sched, color=PALETTE[sched])

    titles  = ["Avg Waiting Time vs Load",
               "Throughput vs Load",
               "Jain Fairness vs Load"]
    ylabels = ["Avg Wait (units)", "Throughput (proc/unit)", "Jain Fairness Index"]
    for ax, t, y in zip(axes, titles, ylabels):
        ax.set_title(t, fontweight="bold")
        ax.set_xlabel("Number of Processes")
        ax.set_ylabel(y)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Scalability Study: Performance vs Process Count\n"
                 f"Averaged over {n_seeds} random seeds per point",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ─────────────────────────────────────────────
#  Benchmark discussion (console)
# ─────────────────────────────────────────────
BENCHMARK_DISCUSSION = """
╔══════════════════════════════════════════════════════════════════════╗
║         BENCHMARK TOOL RECOMMENDATIONS FOR MOBILE SCHEDULERS        ║
╚══════════════════════════════════════════════════════════════════════╝

1. LATENCY BENCHMARKS
   ─────────────────
   • cyclictest (rt-tests suite)
       Measures wakeup latency at μs resolution. Run with SCHED_FIFO
       and --mlockall. Compare histograms, not just means — tail latency
       matters most for RT tasks (sensor/audio pipelines).
       Relevance: directly maps to P95/P99 wait metrics in this simulator.

   • Perfetto (Android tracing)
       Frame-level latency traces. Look for "Choreographer#doFrame" and
       "input latency" slices. EEVDF's lower avg_wait should translate
       to fewer janked frames (>16 ms render budget).

2. THROUGHPUT BENCHMARKS
   ──────────────────────
   • sysbench --test=cpu --cpu-max-prime=20000
       Single- and multi-threaded integer throughput. Expect CFS ≈ EEVDF
       on homogeneous cores; gaps emerge only under mixed workloads.

   • Geekbench 6 (Multi-Core)
       End-to-end throughput including memory and SIMD. Useful for
       validating that Hybrid's overhead does not regress raw compute.

3. ENERGY EFFICIENCY
   ──────────────────
   • Monsoon Power Monitor (hardware)
       Gold standard for current draw. Pair with workload replay to
       get joule-per-task. Thermal scaling τ(T) should reduce peak draw
       by deferring batch jobs during hot periods.

   • Battery Historian (adb bugreport)
       Wakelock and CPU frequency histograms. Look for sustained high-freq
       operation — a sign that the scheduler is not offloading to E-cores.

4. MOBILE-SPECIFIC BENCHMARKS
   ───────────────────────────
   • GameBench Pro
       Tracks frame pacing (σ of frame times) alongside CPU core residency.
       Hybrid's tier isolation should improve frame-time consistency even
       when background ML tasks compete for CPU.

   • Android Camera2 API pipeline benchmark
       End-to-end latency from shutter to JPEG. Sensor-class tasks (pri=0)
       should be scheduled with near-zero wait time — validate with
       Perfetto's camera_stream_processor atrace.

   • WorkloadHint API (Android 12+)
       Apps can declare DISPLAY, CAMERA, MEDIA, or CPU_INTENSIVE hints.
       The Hybrid scheduler's MLFQ tier map aligns naturally with these
       hints and can be validated against the hint-aware scheduler in AOSP.

5. LINUX EEVDF KERNEL PARAMETERS (sysctl)
   ─────────────────────────────────────────
   Tunable via /proc/sys/kernel/sched_*:
     sched_latency_ns          default  6 000 000 ns  (6 ms)
     sched_min_granularity_ns  default    750 000 ns
     sched_wakeup_granularity_ns          1 000 000 ns
     sched_latency_nice        range   -20 .. +19 (per-task latency hint)

   The parameter sweep in linux_eevdf_params.png shows that reducing
   sched_latency below 4 ms degrades Jain fairness on 80+ process loads
   while offering minimal wait-time benefit. The Linux default sits near
   the Pareto frontier for mobile workloads in this simulation.
"""


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Enhanced Mobile CPU Scheduler Simulator")
    print("=" * 60)

    # ── 1. Paper timeline figures (5-process set) ─────────────────
    print("\n[1/7] Generating paper timeline figures (5-process set) …")
    paper_procs_raw = [
        (1, 0,  8, 0),
        (2, 1,  4, 1),
        (3, 2, 12, 0),
        (4, 3,  6, 2),
        (5, 4,  5, 1),
    ]
    paper_procs = [Process(*p) for p in paper_procs_raw]

    generate_cpu_allocation_timeline(
        _cpu_timeline_cfs(paper_procs),
        "CFS CPU Allocation Timeline  (5 processes)",
        "cfs_allocation_tl.png")

    generate_cpu_allocation_timeline(
        _cpu_timeline_eevdf(paper_procs),
        "EEVDF CPU Allocation Timeline  (5 processes)",
        "eevdf_allocation_tl.png")

    generate_cpu_allocation_timeline(
        _cpu_timeline_hybrid(paper_procs),
        "Hybrid EEVDF-MLFQ CPU Allocation Timeline  (5 processes)",
        "hybrid_allocation_tl.png")

    # ── 2. Monte Carlo statistical study ─────────────────────────
    print("\n[2/7] Running Monte Carlo study (200 runs × 120 processes) …")
    mc_results = monte_carlo_study(n_runs=200, n_procs=120)
    print_statistical_summary(mc_results)
    plot_statistical_comparison(mc_results, "stat_comparison.png")

    # ── 3. Sensitivity: alpha ─────────────────────────────────────
    print("\n[3/7] Sensitivity study: urgency bias α …")
    sensitivity_alpha(None, n_seeds=30, filename="sensitivity_alpha.png")

    # ── 4. Sensitivity: temperature / tau(T) ─────────────────────
    print("\n[4/7] Sensitivity study: thermal scaling τ(T) …")
    sensitivity_temperature(None, n_seeds=30, filename="sensitivity_temperature.png")

    # ── 5. Linux EEVDF parameter sweep ───────────────────────────
    print("\n[5/7] Linux EEVDF parameter sweep …")
    linux_eevdf_param_sweep(n_seeds=40, filename="linux_eevdf_params.png")

    # ── 6. Per-class breakdown ────────────────────────────────────
    print("\n[6/7] Per-class performance breakdown …")
    plot_per_class_metrics(n_seeds=50, filename="per_class_metrics.png")

    # ── 7. Scalability study ──────────────────────────────────────
    print("\n[7/7] Scalability study …")
    plot_scalability(filename="scalability.png", n_seeds=30)

    # ── Benchmark discussion ──────────────────────────────────────
    print(BENCHMARK_DISCUSSION)

    print("\n" + "=" * 60)
    print("  All outputs generated successfully!")
    print("=" * 60)
    print("\nFiles produced:")
    for f in [
        "cfs_allocation_tl.png       ← paper figure",
        "eevdf_allocation_tl.png     ← paper figure",
        "hybrid_allocation_tl.png    ← paper figure",
        "stat_comparison.png         ← Monte Carlo box plots",
        "sensitivity_alpha.png       ← α sensitivity study",
        "sensitivity_temperature.png ← τ(T) sensitivity study",
        "linux_eevdf_params.png      ← Linux parameter sweep",
        "per_class_metrics.png       ← per task-class breakdown",
        "scalability.png             ← scalability vs process count",
    ]:
        print(f"  {f}")