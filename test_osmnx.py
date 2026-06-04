import osmnx as ox

print("Testing OSMnx connection...")

try:
    # Try by coordinates instead - centre of Koramangala
    G = ox.graph_from_point(
        (12.9352, 77.6245),   # Koramangala, Bengaluru
        dist=1000,             # 1km radius
        network_type="drive"
    )
    print("SUCCESS!")
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")
except Exception as e:
    print(f"FAILED: {e}")