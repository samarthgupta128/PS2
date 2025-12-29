# 🏎️ Optimized Quarter-Car Suspension Controller

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Algorithm](https://img.shields.io/badge/Algo-RK4_Integration-green?style=for-the-badge)
![Optimization](https://img.shields.io/badge/Optimization-Jerk_Minimization-orange?style=for-the-badge)

> **A high-performance active suspension control system designed to minimize "Comfort Score" for volatile cargo transport.**

---

## 🎯 The Challenge: "Volatile Cargo"
**Stark Industries** requires a suspension controller for an autonomous convoy. The goal is to minimize a weighted Comfort Score ($CS$) based on:
1.  **RMS Displacement** (bouncing)
2.  **Max Displacement** (bottoming out)
3.  **RMS Jerk** (vibration)
4.  **Max Jerk** (sudden shocks)

**Constraints:**
* **Actuator Delay:** 20ms (4 simulation steps)
* **Damping Limits:** 800 - 3500 Ns/m
* **Blind Control:** No preview of the road; only current accelerometer data.

---

## 💡 The Solution: Smoothness First

My analysis revealed that **Max Jerk** dominates the penalty score. Standard "Skyhook" controllers switch damping too fast, creating huge jerk spikes. 

**My approach uses a "Smooth Skyhook" strategy:**

1.  **Predictive Logic**: Uses relative velocity trends to "brace" for impact before it happens.
2.  **Heavy Filtering**: Applies an exponential moving average ($\alpha=0.02$) to damping requests to filter out road noise.
3.  **Strict Rate Limiting**: Caps the rate of change of the damper ($\Delta c / \Delta t$) to prevent sudden force discontinuities.

### The Physics Engine
* **Integration**: Runge-Kutta 4th Order (RK4) for high-precision simulation.
* **Dynamics**: Full 2-DOF Quarter-Car model ($m_s, m_u, k_s, k_t$).

---
