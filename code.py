"""
OPTIMIZED Quarter-Car Active Suspension Controller
===================================================
Goal: MINIMIZE comfort_score = 0.5*rms_zs + max_zs + 0.5*rms_jerk + jerk_max

Key insight: jerk_max dominates the score, so we need:
1. Very smooth damping transitions (minimize jerk)
2. High damping to reduce displacement
3. Heavy filtering to prevent sudden changes
"""

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
M_S = 290.0
M_U = 59.0
K_S = 16_000.0
K_T = 190_000.0
C_MIN = 800.0
C_MAX = 3_500.0
DT = 0.005
DELAY_STEPS = 4


@dataclass
class SimulationResult:
    profile_name: str
    t: np.ndarray
    z_s: np.ndarray
    z_u: np.ndarray
    a_s: np.ndarray
    a_u: np.ndarray
    r: np.ndarray
    c_applied: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# DYNAMICS
# =============================================================================
def quarter_car_derivatives(state: np.ndarray, c: float, r: float) -> np.ndarray:
    z_s, z_s_dot, z_u, z_u_dot = state
    spring_force = K_S * (z_s - z_u)
    damper_force = c * (z_s_dot - z_u_dot)
    tire_force = K_T * (z_u - r)
    z_s_ddot = -(spring_force + damper_force) / M_S
    z_u_ddot = (spring_force + damper_force - tire_force) / M_U
    return np.array([z_s_dot, z_s_ddot, z_u_dot, z_u_ddot])


def rk4_step(state: np.ndarray, c: float, r: float, dt: float) -> np.ndarray:
    k1 = quarter_car_derivatives(state, c, r)
    k2 = quarter_car_derivatives(state + 0.5 * dt * k1, c, r)
    k3 = quarter_car_derivatives(state + 0.5 * dt * k2, c, r)
    k4 = quarter_car_derivatives(state + dt * k3, c, r)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def compute_accelerations(state: np.ndarray, c: float, r: float) -> Tuple[float, float]:
    z_s, z_s_dot, z_u, z_u_dot = state
    spring_force = K_S * (z_s - z_u)
    damper_force = c * (z_s_dot - z_u_dot)
    tire_force = K_T * (z_u - r)
    a_s = -(spring_force + damper_force) / M_S
    a_u = (spring_force + damper_force - tire_force) / M_U
    return float(a_s), float(a_u)


# =============================================================================
# OPTIMIZED CONTROLLER - Focus on minimizing comfort score
# =============================================================================
class OptimizedController:
    """
    Controller optimized to minimize comfort_score.
    
    Key strategies:
    1. Use high damping (near c_max) to minimize displacement
    2. Very heavy smoothing to minimize jerk
    3. Slow rate limiting to prevent sudden changes
    4. Predictive adjustment based on velocity trends
    """
    
    def __init__(self):
        # Base damping - use higher values to reduce displacement
        self.c_base = 2800.0
        
        # Smoothing filter (very aggressive to minimize jerk)
        self.c_filtered = self.c_base
        self.filter_alpha = 0.02  # Very slow filter (heavy smoothing)
        
        # Rate limiter (very tight to minimize jerk_max)
        self.c_prev = self.c_base
        self.max_rate = 500.0  # N·s/m per second (very slow changes)
        
        # State for velocity-based control
        self.prev_z_s_dot = 0.0
        self.prev_dv = 0.0
    
    def __call__(self, z_s: float, z_s_dot: float, 
                 z_u: float, z_u_dot: float,
                 a_s: float, a_u: float) -> float:
        
        dv = z_s_dot - z_u_dot  # Relative velocity
        
        # Skyhook-based damping selection
        if z_s_dot * dv > 0:
            # Body moving away - use high damping
            c_target = 3200.0
        elif abs(z_s_dot) > 0.05:
            # Significant body motion - moderate-high damping
            c_target = 2600.0
        else:
            # Small motion - moderate damping
            c_target = 2000.0
        
        # Add velocity-proportional term (helps reduce displacement)
        c_target += 200.0 * abs(z_s_dot)
        
        # Predict trend and adjust
        dv_rate = (dv - self.prev_dv) / DT if abs(self.prev_dv) > 1e-9 else 0
        self.prev_dv = dv
        
        # If relative velocity is increasing, prepare for impact
        if dv_rate * dv > 0:
            c_target += 100.0
        
        # Clip to valid range
        c_target = np.clip(c_target, C_MIN, C_MAX)
        
        # Heavy low-pass filtering (minimizes jerk)
        self.c_filtered = (1 - self.filter_alpha) * self.c_filtered + self.filter_alpha * c_target
        
        # Strict rate limiting (minimizes jerk_max)
        max_delta = self.max_rate * DT
        c_out = np.clip(self.c_filtered, 
                        self.c_prev - max_delta, 
                        self.c_prev + max_delta)
        self.c_prev = c_out
        
        return float(np.clip(c_out, C_MIN, C_MAX))


class HighDampingController:
    """
    Simple high-damping strategy with maximum smoothing.
    Sometimes simple is best for minimizing jerk.
    """
    
    def __init__(self):
        self.c_nominal = 3000.0  # High constant damping
        self.c_filtered = self.c_nominal
        self.alpha = 0.01  # Very slow filter
        self.c_prev = self.c_nominal
        self.max_rate = 300.0  # Very slow rate limit
    
    def __call__(self, z_s: float, z_s_dot: float, 
                 z_u: float, z_u_dot: float,
                 a_s: float, a_u: float) -> float:
        
        dv = z_s_dot - z_u_dot
        
        # Simple skyhook with minimal variation
        if z_s_dot * dv > 0:
            c_target = 3300.0
        else:
            c_target = 2700.0
        
        # Very heavy smoothing
        self.c_filtered = (1 - self.alpha) * self.c_filtered + self.alpha * c_target
        
        # Strict rate limit
        max_delta = self.max_rate * DT
        c_out = np.clip(self.c_filtered, self.c_prev - max_delta, self.c_prev + max_delta)
        self.c_prev = c_out
        
        return float(np.clip(c_out, C_MIN, C_MAX))


class UltraSmoothController:
    """
    Ultra-smooth controller - prioritizes jerk minimization above all.
    Uses nearly constant damping with minimal variation.
    """
    
    def __init__(self):
        # Near-constant high damping
        self.c_base = 2900.0
        self.c_current = self.c_base
        self.alpha = 0.005  # Extremely slow adaptation
        self.max_rate = 200.0  # Extremely slow rate limit
    
    def __call__(self, z_s: float, z_s_dot: float, 
                 z_u: float, z_u_dot: float,
                 a_s: float, a_u: float) -> float:
        
        dv = z_s_dot - z_u_dot
        
        # Tiny adjustments only
        if z_s_dot * dv > 0 and abs(z_s_dot) > 0.02:
            c_target = self.c_base + 200
        elif z_s_dot * dv < 0 and abs(z_s_dot) > 0.02:
            c_target = self.c_base - 200
        else:
            c_target = self.c_base
        
        # Extremely slow filter
        c_target = np.clip(c_target, C_MIN, C_MAX)
        self.c_current = (1 - self.alpha) * self.c_current + self.alpha * c_target
        
        # Strict rate limit
        max_delta = self.max_rate * DT
        c_out = np.clip(self.c_current, self.c_current - max_delta, self.c_current + max_delta)
        
        return float(np.clip(c_out, C_MIN, C_MAX))


class AdaptiveTunedController:
    """
    Adaptive controller with parameters tuned per profile type.
    """
    
    def __init__(self):
        self.c_filtered = 2500.0
        self.c_prev = 2500.0
        
        # Adaptive parameters
        self.roughness = 0.0
        self.roughness_alpha = 0.01
        
    def __call__(self, z_s: float, z_s_dot: float, 
                 z_u: float, z_u_dot: float,
                 a_s: float, a_u: float) -> float:
        
        # Estimate road roughness
        self.roughness = (1 - self.roughness_alpha) * self.roughness + self.roughness_alpha * abs(a_u)
        
        dv = z_s_dot - z_u_dot
        
        # Adaptive base damping based on roughness
        if self.roughness < 5:
            c_base = 2800.0  # Smooth road - high comfort damping
            alpha = 0.015
            max_rate = 400.0
        elif self.roughness < 15:
            c_base = 2600.0  # Medium road
            alpha = 0.02
            max_rate = 600.0
        else:
            c_base = 2400.0  # Rough road - more responsive
            alpha = 0.03
            max_rate = 800.0
        
        # Skyhook logic
        if z_s_dot * dv > 0:
            c_target = c_base + 400
        else:
            c_target = c_base - 200
        
        c_target = np.clip(c_target, C_MIN, C_MAX)
        
        # Smooth
        self.c_filtered = (1 - alpha) * self.c_filtered + alpha * c_target
        
        # Rate limit
        max_delta = max_rate * DT
        c_out = np.clip(self.c_filtered, self.c_prev - max_delta, self.c_prev + max_delta)
        self.c_prev = c_out
        
        return float(np.clip(c_out, C_MIN, C_MAX))


# =============================================================================
# SIMULATION
# =============================================================================
def simulate_profile(t: np.ndarray, r: np.ndarray, 
                     profile_name: str, controller) -> SimulationResult:
    n = len(t)
    state = np.zeros(4)
    delay_buffer = deque([2500.0] * DELAY_STEPS, maxlen=DELAY_STEPS)
    
    z_s_hist = np.zeros(n)
    z_u_hist = np.zeros(n)
    a_s_hist = np.zeros(n)
    a_u_hist = np.zeros(n)
    c_hist = np.zeros(n)
    
    for i in range(n):
        c_applied = delay_buffer[0]
        r_i = r[i]
        
        state = rk4_step(state, c_applied, r_i, DT)
        a_s, a_u = compute_accelerations(state, c_applied, r_i)
        
        z_s_hist[i] = state[0]
        z_u_hist[i] = state[2]
        a_s_hist[i] = a_s
        a_u_hist[i] = a_u
        c_hist[i] = c_applied
        
        c_request = controller(state[0], state[1], state[2], state[3], a_s, a_u)
        delay_buffer.append(c_request)
    
    result = SimulationResult(
        profile_name=profile_name, t=t,
        z_s=z_s_hist, z_u=z_u_hist,
        a_s=a_s_hist, a_u=a_u_hist,
        r=r, c_applied=c_hist
    )
    result.metrics = compute_metrics(z_s_hist, a_s_hist)
    return result


def compute_metrics(z_s: np.ndarray, a_s: np.ndarray) -> Dict[str, float]:
    z_s_rel = z_s - z_s[0]
    rms_zs = float(np.sqrt(np.mean(z_s_rel ** 2)))
    max_zs = float(np.max(np.abs(z_s_rel)))
    
    jerk = np.diff(a_s) / DT
    rms_jerk = float(np.sqrt(np.mean(jerk ** 2)))
    jerk_max = float(np.max(np.abs(jerk)))
    
    comfort_score = 0.5 * rms_zs + max_zs + 0.5 * rms_jerk + jerk_max
    
    return {
        "rms_zs": rms_zs,
        "max_zs": max_zs,
        "rms_jerk": rms_jerk,
        "jerk_max": jerk_max,
        "comfort_score": comfort_score
    }


def plot_results(result: SimulationResult, output_dir: str = "plots_optimized") -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(result.t, result.r * 1000, 'k--', label='Road', alpha=0.5)
    ax1.plot(result.t, result.z_u * 1000, 'b-', label='Wheel z_u', alpha=0.7)
    ax1.plot(result.t, result.z_s * 1000, 'r-', label='Body z_s', linewidth=1.5)
    ax1.set_ylabel('Displacement (mm)')
    ax1.set_title(f'{result.profile_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(result.t, result.a_s, 'r-', label='Body accel', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    m = result.metrics
    fig.text(0.02, 0.02, 
             f"rms_zs={m['rms_zs']*1000:.3f}mm, max_zs={m['max_zs']*1000:.3f}mm, "
             f"rms_jerk={m['rms_jerk']:.2f}, jerk_max={m['jerk_max']:.2f}, comfort={m['comfort_score']:.2f}",
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{result.profile_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


# =============================================================================
# MAIN - TEST ALL CONTROLLERS AND PICK BEST
# =============================================================================
def load_road_profiles(csv_path: str = "road_profiles.csv"):
    df = pd.read_csv(csv_path)
    t = df['t'].to_numpy()
    profiles = {}
    for i in range(1, 6):
        profiles[f'profile_{i}'] = df[f'profile_{i}'].to_numpy()
    return t, profiles


def main():
    print("=" * 70)
    print("OPTIMIZED Controller - Minimizing Comfort Score")
    print("=" * 70)
    
    t, profiles = load_road_profiles()
    
    controllers = {
        "Optimized": OptimizedController,
        "HighDamping": HighDampingController,
        "UltraSmooth": UltraSmoothController,
        "Adaptive": AdaptiveTunedController,
    }
    
    # Test each controller on each profile
    print("\nTesting all controllers on all profiles...")
    all_results = {}
    
    for ctrl_name, ctrl_class in controllers.items():
        all_results[ctrl_name] = {}
        total_comfort = 0
        for profile_name, r in profiles.items():
            controller = ctrl_class()
            result = simulate_profile(t, r, profile_name, controller)
            all_results[ctrl_name][profile_name] = result
            total_comfort += result.metrics['comfort_score']
        print(f"  {ctrl_name}: total_comfort = {total_comfort:.2f}")
    
    # Find best controller
    best_ctrl = min(all_results.keys(), 
                    key=lambda c: sum(all_results[c][p].metrics['comfort_score'] 
                                      for p in profiles.keys()))
    
    print(f"\n*** Best controller: {best_ctrl} ***")
    
    # Run final simulation with best controller
    print(f"\nFinal results with {best_ctrl}:")
    print("-" * 70)
    
    final_results = []
    rows = []
    
    for profile_name, r in profiles.items():
        controller = controllers[best_ctrl]()
        result = simulate_profile(t, r, profile_name, controller)
        final_results.append(result)
        
        m = result.metrics
        print(f"{profile_name}: rms_zs={m['rms_zs']:.6f}, max_zs={m['max_zs']:.6f}, "
              f"rms_jerk={m['rms_jerk']:.2f}, comfort={m['comfort_score']:.2f}")
        
        rows.append({
            "profile": profile_name,
            "rms_zs": m["rms_zs"],
            "max_zs": m["max_zs"],
            "rms_jerk": m["rms_jerk"],
            "comfort_score": m["comfort_score"]
        })
        
        plot_results(result)
    
    # Save submission
    df = pd.DataFrame(rows)
    df = df[["profile", "rms_zs", "max_zs", "rms_jerk", "comfort_score"]]
    df.to_csv("submission.csv", index=False)
    
    total = sum(r.metrics['comfort_score'] for r in final_results)
    print("-" * 70)
    print(f"TOTAL comfort score: {total:.2f}")
    print("\nSaved: submission.csv, plots_optimized/*.png")


if __name__ == "__main__":
    main()
