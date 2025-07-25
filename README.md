# MSCI434-FinalProject

# Train Blocking Optimization — MSCI 434 Final Project

This project implements and extends a two-stage optimization model for **joint train blocking and shipment path planning**, replicating the methodology presented in the research paper _"Joint Optimization of Train Blocking and Shipment Path"_.

Developed for the **MSCI 434: Decision Models in Supply Chain Management** course, the goal of this project is to understand and apply real-world supply chain optimization methods using Python and Gurobi.

---

## Project Objectives

- Replicate the two-stage optimization model from a published journal paper.
- Use Gurobi to solve both the **shipment path** and **train blocking** subproblems.
- Analyze how randomized demand affects solution patterns and costs.
- Implement a method to generate an **initial feasible solution** (warm start) and compare performance.
- Propose and implement a **custom model extension** to better reflect real-world behavior.
- Evaluate yard usage frequency and shipment path trends across multiple trials.

---

## Problem Overview

### 1. **Shipment Path Model**
- **Goal:** Determine optimal shipment paths for OD pairs that minimize total car-kilometers.
- **Constraints:** Flow conservation and logical routing through the rail network.
- **Output:** A unique path for each OD pair.

### 2. **Train Blocking Model**
- **Goal:** Assign classification yards to intermediate points on each path to minimize total blocking cost.
- **Constraints:**
  - Only one yard per shipment path
  - Yard capacity limits
  - Binary decision variables for blocking and building blocks

---

## Extensions Implemented

### 1. **Initial Feasible Solution (Warm Start)**
- After solving the blocking model once, a feasible solution is extracted and used to warm-start Gurobi.
- This approach helps reduce computation time in subsequent solves.
- Comparison of solve times with and without the initial solution is logged.

### 2. **Distance Penalty Extension**
- A soft penalty was added for selecting classification yards that deviate too far from the shortest OD path.
- This reflects real-world routing efficiency constraints and encourages more logical yard assignments.
- The extended model shows a trade-off between routing flexibility and blocking efficiency.

---

## Experiments

- Conducted **20 randomized trials**, varying OD demands using a Gaussian perturbation.
- For each trial:
  - Solved shipment path model
  - Solved train blocking model (with strict yard capacity)
- Extracted total cost, yard assignment patterns, and block builds.
- Identified and visualized:
  - Most frequently used yards
  - Most common shipment paths
  - Top 10 cost-efficient solutions

---

## How It Was Achieved

- All optimization models were built and solved using **Python and Gurobi**.
- Data was loaded from Excel and converted into a graph using **NetworkX**.
- Paths and blocks were computed and stored for each trial.
- Performance and cost metrics were visualized using **Matplotlib** and **Seaborn**.
- Top-performing results were exported for reproducibility and analysis.

---

## Key Outputs

- `top_yard_usage.csv`: Yards most frequently chosen as classification points.
- `top_shipment_paths.csv`: Common shipment routing paths across top trials.
- `top_trials_summary.json`: Full detail of top trial paths and assignments.

---


