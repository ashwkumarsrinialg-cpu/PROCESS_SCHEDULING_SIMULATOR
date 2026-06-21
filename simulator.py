import matplotlib.pyplot as plt
import numpy as np

class Process:
    def __init__(self, pid, arrival, burst, priority=0):
        self.pid = pid
        self.arrival = arrival
        self.burst = burst
        self.remaining = burst
        self.priority = priority  # Lower number = higher priority
        self.start_time = -1
        self.completion_time = -1
        self.vruntime = 0.0
        self.deadline = 0.0


def simulate_cfs_cpu_allocation(processes):
    """Simulate CFS and generate CPU allocation timeline data"""
    time = 0
    cpu_timeline = []
    active = []
    completed = 0
    n = len(processes)
    proc_list = [Process(*p) for p in processes]
    
    while completed < n:
        # Add newly arrived processes
        for p in proc_list:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                active.append(p)
        
        if not active:
            time += 1
            continue
            
        # Select process with smallest vruntime (CFS logic)
        current = min(active, key=lambda p: p.vruntime)
        if current.start_time == -1:
            current.start_time = time
        
        slice_time = min(2, current.remaining)
        cpu_timeline.append((current.pid, time, time + slice_time))
        
        current.remaining -= slice_time
        current.vruntime += slice_time * (1.0 / (1 + current.priority))
        time += slice_time
        
        if current.remaining <= 0:
            current.completion_time = time
            completed += 1
            active.remove(current)
    
    return cpu_timeline


def simulate_eevdf_cpu_allocation(processes):
    """Simulate EEVDF and generate CPU allocation timeline data"""
    time = 0
    cpu_timeline = []
    active = []
    completed = 0
    n = len(processes)
    proc_list = [Process(*p) for p in processes]
    
    while completed < n:
        for p in proc_list:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                active.append(p)
                p.deadline = p.arrival + p.burst / (1 + p.priority)
        
        if not active:
            time += 1
            continue
            
        # Earliest virtual deadline (EEVDF logic)
        current = min(active, key=lambda p: p.deadline)
        if current.start_time == -1:
            current.start_time = time
        
        slice_time = min(2, current.remaining)
        cpu_timeline.append((current.pid, time, time + slice_time))
        
        current.remaining -= slice_time
        time += slice_time
        
        current.deadline = time + current.remaining / (1 + current.priority)
        
        if current.remaining <= 0:
            current.completion_time = time
            completed += 1
            active.remove(current)
    
    return cpu_timeline


def simulate_hybrid_cpu_allocation(processes):
    """Simulate Hybrid (MLFQ + EEVDF) and generate CPU allocation timeline data"""
    time = 0
    cpu_timeline = []
    active = []
    completed = 0
    n = len(processes)
    proc_list = [Process(*p) for p in processes]
    
    while completed < n:
        for p in proc_list:
            if p.arrival <= time and p.remaining > 0 and p not in active:
                active.append(p)
        
        if not active:
            time += 1
            continue
            
        # MLFQ: Highest priority class
        min_prio = min(p.priority for p in active)
        candidates = [p for p in active if p.priority == min_prio]
        
        # EEVDF within the class
        current = min(candidates, key=lambda p: p.arrival + p.burst / (1 + p.priority))
        
        if current.start_time == -1:
            current.start_time = time
        
        slice_time = min(2, current.remaining)
        cpu_timeline.append((current.pid, time, time + slice_time))
        
        current.remaining -= slice_time
        time += slice_time
        
        if current.remaining <= 0:
            current.completion_time = time
            completed += 1
            active.remove(current)
    
    return cpu_timeline


def generate_cpu_allocation_timeline(cpu_timeline, title, filename):
    """Generate and save CPU Allocation Timeline visualization"""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (pid, start, end) in enumerate(cpu_timeline):
        ax.barh(f'P{pid}', end - start, left=start, 
                color=colors[i % 10], edgecolor='black', height=0.6)
        ax.text(start + (end - start)/2, f'P{pid}', f'P{pid}', 
                ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_xlabel('Time Units')
    ax.set_ylabel('Process')
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("=== CPU Allocation Timeline Output Generating Code ===\n")
    
    # Process format: (pid, arrival_time, burst_time, priority)
    processes = [
        (1, 0, 8, 0),
        (2, 1, 4, 1),
        (3, 2, 12, 0),
        (4, 3, 6, 2),
        (5, 4, 5, 1)
    ]
    
    # Generate CPU allocation timelines
    cfs_tl = simulate_cfs_cpu_allocation(processes)
    eevdf_tl = simulate_eevdf_cpu_allocation(processes)
    hybrid_tl = simulate_hybrid_cpu_allocation(processes)
    
    # Save timeline charts
    generate_cpu_allocation_timeline(cfs_tl, 'CFS CPU Allocation Timeline', 'cfs_allocation_tl.png')
    generate_cpu_allocation_timeline(eevdf_tl, 'EEVDF CPU Allocation Timeline', 'eevdf_allocation_tl.png')
    generate_cpu_allocation_timeline(hybrid_tl, 'Hybrid CPU Allocation Timeline', 'hybrid_allocation_tl.png')
    
    print("✅ CPU Allocation Timeline charts generated successfully!")
    print("   - cfs_allocation_tl.png")
    print("   - eevdf_allocation_tl.png")
    print("   - hybrid_allocation_tl.png")