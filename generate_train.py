import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
import joblib

# ----------------------------
# Process class
# ----------------------------
class Process:
    def __init__(self, pid, burst):
        self.pid = pid
        self.burst_time = burst
        self.remaining_time = burst
        self.waiting_time = 0

# ----------------------------
# Round Robin simulation
# ----------------------------
def simulate_rr(processes, quantum):
    processes = deepcopy(processes)
    timeline = []
    time = 0
    ready_queue = [p for p in processes if p.remaining_time > 0]

    while any(p.remaining_time > 0 for p in processes):
        for p in ready_queue:
            run_time = min(quantum, p.remaining_time)
            p.remaining_time -= run_time
            timeline.extend([p.pid]*run_time)
            for other in ready_queue:
                if other != p and other.remaining_time > 0:
                    other.waiting_time += run_time
            time += run_time
        ready_queue = [p for p in processes if p.remaining_time > 0]

    avg_waiting = np.mean([p.waiting_time for p in processes])
    cpu_util = 100 * len([x for x in timeline if x>0]) / max(1, len(timeline))
    return avg_waiting, cpu_util

# ----------------------------
# Generate dataset
# ----------------------------
num_snapshots = 5000  # Increase for better ML model
snapshots = []

print("Generating dataset...")
for _ in range(num_snapshots):
    num_procs = np.random.randint(3, 11)  # 3–10 processes
    bursts = np.random.randint(1, 11, size=num_procs)  # burst times 1–10
    processes = [Process(i+1, bursts[i]) for i in range(num_procs)]

    num_ready = num_procs
    avg_remaining = np.mean(bursts)
    max_remaining = np.max(bursts)
    avg_waiting = 0
    cpu_util = 0

    # Find optimal quantum 1–10
    best_quantum = 1
    best_waiting = float('inf')
    for q in range(1, 11):
        avg_w, _ = simulate_rr(processes, q)
        if avg_w < best_waiting:
            best_waiting = avg_w
            best_quantum = q

    snapshots.append([num_ready, avg_remaining, max_remaining, avg_waiting, cpu_util, best_quantum])

# Save dataset
df = pd.DataFrame(snapshots, columns=["num_ready","avg_remaining","max_remaining","avg_waiting","cpu_util","optimal_quantum"])
df.to_csv("quantum_dataset.csv", index=False)
print("✅ Dataset saved as quantum_dataset.csv")

# ----------------------------
# Train ML model
# ----------------------------
X = df[["num_ready","avg_remaining","max_remaining","avg_waiting","cpu_util"]]
y = df["optimal_quantum"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "dynamic_quantum_model.pkl")
print("✅ ML model saved as dynamic_quantum_model.pkl")
