def visualize_graph_pyg(data, initial_nodes, terminating_nodes):
    G = nx.DiGraph()
    for src, dest in data.edge_index.t().tolist():
        G.add_edge(src, dest)
    
    plt.figure(figsize=(20, 15))  # Increased figure size for better spacing
    # Use the Kamada-Kawai layout for potentially better distribution of nodes
    pos = nx.kamada_kawai_layout(G)
    
    # Draw all nodes and edges with basic styling
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    all_nodes = set(G.nodes())
    middle_nodes = all_nodes - initial_nodes - terminating_nodes

    # Draw initial nodes in green, terminating nodes in red, and middle nodes in skyblue
    nx.draw_networkx_nodes(G, pos, nodelist=initial_nodes, node_color='green', label="Initial Nodes", node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=terminating_nodes, node_color='red', label="Terminating Nodes", node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=middle_nodes, node_color='skyblue', node_size=500)
    
    # Optional: Draw labels for only initial and terminating nodes to reduce clutter
    labels = {n: n for n in G.nodes() if n in initial_nodes or n in terminating_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.legend(scatterpoints=1)
    plt.title("Causality Graph with Initial and Terminating Nodes")
    plt.axis('off')  # Turn off the axis
    plt.show()