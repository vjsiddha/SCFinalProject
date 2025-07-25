import pandas as pd
import networkx as nx
from gurobipy import Model, GRB, quicksum

# ------------------------------------------------------------------
# Data Loaders
# ------------------------------------------------------------------
def load_yard_parameters(filepath):
    df = pd.read_excel(filepath)
    yard_info = {}
    for _, row in df.iterrows():
        yard = row['Yard']
        yard_info[yard] = {
            'reclass_delay': row['RC'],
            'capacity': row['COST'],
            'sort_tracks': row['ST'],
            'beta': row['AP']
        }
    return yard_info

def load_link_parameters(filepath):
    df = pd.read_excel(filepath)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        i, j = row['NodeA'], row['NodeB']
        G.add_edge(i, j, length=row['Length'], capacity=row['CapaAB'], alpha=1.0)
    return G

def load_od_demands(filepath):
    df = pd.read_excel(filepath, index_col=0)
    od_pairs = []
    for origin in df.index:
        for destination in df.columns:
            cars = df.loc[origin, destination]
            if cars > 0 and origin != destination:
                od_pairs.append((origin, destination, cars))
    return od_pairs

# ------------------------------------------------------------------
# Shortest Paths
# ------------------------------------------------------------------
def compute_shortest_paths(G, od_pairs):
    shortest_paths = {}
    shortest_lengths = {}
    for o, d, _ in od_pairs:
        try:
            path = nx.shortest_path(G, source=o, target=d, weight='length')
            length = nx.shortest_path_length(G, source=o, target=d, weight='length')
            shortest_paths[(o, d)] = path
            shortest_lengths[(o, d)] = length
        except nx.NetworkXNoPath:
            print(f"⚠️ No path from {o} to {d}")
    return shortest_paths, shortest_lengths

# ------------------------------------------------------------------
# Shipment Path Subproblem
# ------------------------------------------------------------------
def solve_path_model(G, od_pairs, shortest_lengths, epsilon=1.2, train_capacity=1e9):
    model = Model("ShipmentPath")
    model.setParam('OutputFlag', 0)
    x = {}
    for o, d, n in od_pairs:
        for i, j in G.edges():
            x[o, d, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{o}_{d}_{i}_{j}")
    model.update()

    model.setObjective(
        quicksum(n * G[i][j]['length'] * x[o, d, i, j] for (o, d, n) in od_pairs for (i, j) in G.edges()),
        GRB.MINIMIZE
    )

    for o, d, n in od_pairs:
        for k in G.nodes():
            inflow = quicksum(x[o, d, i, k] for i in G.predecessors(k))
            outflow = quicksum(x[o, d, k, j] for j in G.successors(k))
            if k == o:
                model.addConstr(outflow - inflow == 1)
            elif k == d:
                model.addConstr(outflow - inflow == -1)
            else:
                model.addConstr(outflow - inflow == 0)

    for o, d, n in od_pairs:
        shortest_dist = shortest_lengths[(o, d)]
        model.addConstr(quicksum(G[i][j]['length'] * x[o, d, i, j] for (i, j) in G.edges()) <= epsilon * shortest_dist)

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        paths = {}
        for o, d, _ in od_pairs:
            edges_used = [(i, j) for (i, j) in G.edges() if x[o, d, i, j].X > 0.5]
            paths[(o, d)] = edges_used
        return model.ObjVal, paths
    else:
        raise RuntimeError("❌ Gurobi did not find an optimal solution.")

# ------------------------------------------------------------------
# Train Blocking Subproblem with Distance Penalty Extension
# ------------------------------------------------------------------
def solve_block_model_with_distance_penalty(
    yard_info, od_pairs, selected_paths, shortest_lengths,
    penalty_factor=0.05,  # controls strength of distance penalty
    strict_capacity=True,
    initial_solution=None
):
    from gurobipy import Model, GRB, quicksum

    model = Model("TrainBlockingWithDistancePenalty")
    model.setParam('OutputFlag', 0)
    z, y = {}, {}

    # z[o,d,k] = 1 if OD pair (o,d) is classified at yard k
    for (o, d, n) in od_pairs:
        path_nodes = {i for i, _ in selected_paths[(o, d)]} | {j for _, j in selected_paths[(o, d)]}
        intermediate_nodes = sorted(k for k in path_nodes if k in yard_info)
        if intermediate_nodes:
            for k in intermediate_nodes:
                z[o, d, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{o}_{d}_{k}")
                if initial_solution and (o, d, k) in initial_solution:
                    z[o, d, k].start = initial_solution[(o, d, k)]

    # y[k,d] = 1 if yard k builds a block for destination d
    dests = {d for _, d, _ in od_pairs}
    for k in yard_info:
        for d in dests:
            y[k, d] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{d}")
            if initial_solution and ("y", k, d) in initial_solution:
                y[k, d].start = initial_solution[("y", k, d)]

    model.update()

    # --- Distance Penalty Extension ---
    distance_penalty = {}
    for (o, d, n) in od_pairs:
        for k in yard_info:
            if (o, d, k) in z:
                try:
                    extra = (shortest_lengths[(o, k)] + shortest_lengths[(k, d)]) - shortest_lengths[(o, d)]
                except KeyError:
                    extra = 999999  # fallback: very high penalty if no path exists
                distance_penalty[o, d, k] = max(0, extra)

    # Objective with distance penalty
    model.setObjective(
        quicksum(
            n * yard_info[k]['reclass_delay'] * z[o, d, k] +
            penalty_factor * distance_penalty[o, d, k] * z[o, d, k]
            for (o, d, n) in od_pairs for k in yard_info if (o, d, k) in z
        ), GRB.MINIMIZE
    )

    # Constraint 1: Each OD pair classified at one yard
    for o, d, n in od_pairs:
        cand_ks = [k for k in yard_info if (o, d, k) in z]
        if cand_ks:
            model.addConstr(quicksum(z[o, d, k] for k in cand_ks) == 1)

    # Constraint 2: OD pair uses a yard only if the yard builds a block
    for (o, d, n) in od_pairs:
        for k in yard_info:
            if (o, d, k) in z:
                model.addConstr(z[o, d, k] <= y[k, d])

    # Constraint 3: Capacity constraint per yard
    for k in yard_info:
        model.addConstr(quicksum(y[k, d] for d in dests) <= yard_info[k]['capacity'])

    model.optimize()

    if model.status == GRB.OPTIMAL:
        block_assignments = {(o, d): [] for o, d, _ in od_pairs}
        blocks_built = []

        for (o, d, k), var in z.items():
            if var.X > 0.5:
                block_assignments[(o, d)].append(k)

        for (k, d), var in y.items():
            if var.X > 0.5:
                blocks_built.append((k, d))

        print("\n🔍 Classification yard assignment (with distance penalty):")
        for (o, d, _) in od_pairs:
            found = False
            for k in yard_info:
                if (o, d, k) in z and z[o, d, k].X > 0.5:
                    print(f"  - {o} → {d} classified at yard {k}")
                    found = True
                    break
            if not found:
                print(f"  - {o} → {d} ❌ NO classification yard assigned")

        print("\n📊 Yard capacity usage:")
        for k in yard_info:
            used = sum(1 for d in dests if y[k, d].X > 0.5)
            max_cap = yard_info[k]['capacity']
            print(f"   - Yard {k}: used {used} / capacity {max_cap}")
            if used > max_cap:
                print(f"     ⚠️ WARNING: Yard {k} exceeds its capacity!")

        return model.objVal, block_assignments, blocks_built, None

    else:
        raise RuntimeError("❌ Gurobi did not find an optimal solution for the blocking model.")
# ------------------------------------------------------------------
# Train Blocking Subproblem
# ------------------------------------------------------------------

    model.setObjective(
        quicksum(n * yard_info[k]['reclass_delay'] * z[o, d, k]
                 for (o, d, n) in od_pairs for k in yard_info if (o, d, k) in z),
        GRB.MINIMIZE
    )

    for o, d, n in od_pairs:
        cand_ks = [k for k in yard_info if (o, d, k) in z]
        if cand_ks:
            model.addConstr(quicksum(z[o, d, k] for k in cand_ks) == 1)

    for (o, d, n) in od_pairs:
        for k in yard_info:
            if (o, d, k) in z:
                model.addConstr(z[o, d, k] <= y[k, d])

    for k in yard_info:
        cap = yard_info[k]['capacity']
        if not strict_capacity and cap < 5:
            print(f"⚠️ Warning: Yard {k} has low capacity ({cap}). Using relaxed limit of 1000.")
            cap = 1000
        
    model.optimize()

    if model.status == GRB.OPTIMAL:
        block_assignments = {(o, d): [] for o, d, _ in od_pairs}
        blocks_built = []

        for (o, d, k), var in z.items():
            if var.X > 0.5:
                block_assignments[(o, d)].append(k)

        for (k, d), var in y.items():
            if var.X > 0.5:
                blocks_built.append((k, d))

        print("\n🔍 DEBUG: Classification yard assignment for each OD pair")
        for (o, d, _) in od_pairs:
            found = False
            for k in yard_info:
                if (o, d, k) in z and z[o, d, k].X > 0.5:
                    print(f"  - {o} → {d} classified at yard {k}")
                    found = True
                    break
            if not found:
                print(f"  - {o} → {d} ❌ NO classification yard assigned")

        print("\n⚠️ OD pairs with no classification:")
        for (o, d), yards in block_assignments.items():
            if not yards:
                print(f"   - {o} → {d}")

        print("\n🏭 Yard capacity usage:")
        for k in yard_info:
            used = sum(1 for d in dests if y[k, d].X > 0.5)
            max_cap = yard_info[k]['capacity']
            print(f"   - Yard {k}: used {used} / capacity {max_cap}")
            if used > max_cap:
                print(f"     ⚠️ WARNING: Yard {k} exceeds its capacity!")

        return model.objVal, block_assignments, blocks_built
    else:
        raise RuntimeError("❌ Gurobi did not find an optimal solution for the blocking model.")