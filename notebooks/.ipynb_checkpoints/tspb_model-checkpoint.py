import pandas as pd
import networkx as nx

def load_yard_parameters(filepath):
    df = pd.read_excel(filepath)
    yard_info = {}
    for _, row in df.iterrows():
        yard = row['Yard']
        yard_info[yard] = {
            'reclass_delay': row['ti'],
            'capacity': row['gi'],
            'sort_tracks': row['hi'],
            'beta': row['βi']
        }
    return yard_info

def load_link_parameters(filepath):
    df = pd.read_excel(filepath)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        i, j = row['From'], row['To']
        G.add_edge(i, j, 
                   length=row['li,j'], 
                   capacity=row['fi,j'],
                   alpha=row['αi,j'])
    return G

def load_od_demands(filepath):
    df = pd.read_excel(filepath)
    od_pairs = []
    for _, row in df.iterrows():
        od_pairs.append((row['Origin'], row['Destination'], row['no,d']))
    return od_pairs

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
            print(f"No path from {o} to {d}")
    return shortest_paths, shortest_lengths
