import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt



"""
Reads messages from a specified file and organizes them into sections.
    Parameters:
    - file_path: Path to the file containing the messages.
    
    Returns:
    - messages: A list of messages, each represented as a tuple (index, src, dest).
    - sections: A list of sections, each containing a list of message strings.
    
"""
def read_messages_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip().split('#')
    sections = [section.strip().split('\n') for section in content if section.strip()]
    messages = []
    for section in sections:
        for line in section:
            if line:
                parts = line.split(':')
                index, src, dest = int(parts[0].strip()), parts[1].strip(), parts[2].strip()
                messages.append((index, src, dest))
    return messages, sections


"""
Constructs a causality graph from the given messages.
    
    Parameters:
    - messages: A list of messages, each represented as a tuple (index, src, dest).
    - sections: A list of sections from the file, used to determine initial and terminating nodes.
    
    Returns:
    - A Data object representing the graph, sets of initial and terminating nodes, and a dictionary of successors.
    """

def construct_causality_graph(messages, sections):
    edge_list = []
    initial_nodes = set(int(line.split(':')[0].strip()) for line in sections[0] if line)
    terminating_nodes = set(int(line.split(':')[0].strip()) for line in sections[-1] if line)
    successors_dict = {}  # Adjusted to hold more detailed info
    
    for index, src, dest in messages:
        if index not in successors_dict:
            successors_dict[index] = {'src': src, 'dest': dest, 'successors': []}
        for _, s, d in messages:
            if dest == s and index not in terminating_nodes and _ not in initial_nodes:
                edge_list.append((index, _))
                successors_dict[index]['successors'].append(_)
                if _ not in successors_dict:
                    successors_dict[_] = {'src': s, 'dest': d, 'successors': []}

    edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long)
    return Data(edge_index=edge_index), initial_nodes, terminating_nodes, successors_dict


def get_successors(node_index, successors_dict):
    node_info = successors_dict.get(node_index)
    if node_info:
        successors_info = [(succ, successors_dict[succ]['src'], successors_dict[succ]['dest']) for succ in node_info['successors']]
        return successors_info
    return []

"""
Prints detailed path information for the successors of the specified node.
    
    Parameters:
    - node_index: The index of the node to query.
    - successors_dict: A dictionary mapping each node to its details and list of successors.
"""

def print_successor_info(node_index, successors_dict):

    if node_index in successors_dict:
        node_info = successors_dict[node_index]
        print(f"Node {node_index} (Source: {node_info['src']}, Destination: {node_info['dest']}) has successors:")
        for succ in node_info['successors']:
            succ_info = successors_dict.get(succ, {})
            print(f"  Successor: {succ}, Path: {succ_info.get('src', 'N/A')} -> {succ_info.get('dest', 'N/A')}")
    else:
        print(f"Node {node_index} has no successors or does not exist in the graph.")


"""
Visualizes the causality graph using NetworkX and Matplotlib.
    
    Parameters:
    - data: A Data object representing the graph.
    - initial_nodes: A set of initial node indices.
    - terminating_nodes: A set of terminating node indices.
"""

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
    nx.draw_networkx_nodes(G, pos, nodelist=middle_nodes, node_color='skyblue', node_size=600)
    
    # Optional: Draw labels for only initial and terminating nodes to reduce clutter
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.legend(scatterpoints=1)
    plt.title("Causality Graph with Initial and Terminating Nodes")
    plt.axis('off')  # Turn off the axis
    plt.show()

"""
pass in two indices, and check if there is a connection in the graph.
"""
def isCausal(node_index1, node_index2, successors_dict):
    successors_of_node1 = successors_dict.get(node_index1, {}).get('successors', [])
    successors_of_node2 = successors_dict.get(node_index2, {}).get('successors', [])
    
    if node_index1 in successors_of_node2 or node_index2 in successors_of_node1:
        return True
    return False

"""
print if 2 nodes are connected or not.
"""
def print_causality_info(node_index1, node_index2, successors_dict):
    successors_of_node1 = successors_dict.get(node_index1, {}).get('successors', [])
    successors_of_node2 = successors_dict.get(node_index2, {}).get('successors', [])
    
    if node_index1 in successors_of_node2:
        print(f"Node {node_index2} is causal to Node {node_index1}.")
    elif node_index2 in successors_of_node1:
        print(f"Node {node_index1} is causal to Node {node_index2}.")
    else:
        print(f"Nodes {node_index1} and {node_index2} are not connected.")

if __name__ == "__main__":
    file_path = "graph/newLarge.msg"  
    messages, sections = read_messages_from_file(file_path)
    data, initial_nodes, terminating_nodes, successors_dict = construct_causality_graph(messages, sections)
    
    # Print detailed path information for a specific node
    node_index_to_query = 0  # Change this to the node index you're interested in
    print_successor_info(node_index_to_query, successors_dict)
    # print_successor_info(7, successors_dict)
    # print_successor_info(25, successors_dict)

    #see if causal
    node1 = 0
    node2 = 25
    print_causality_info(node1, node2, successors_dict)
    print_causality_info(0, 7, successors_dict)

    # Visualize the graph
    visualize_graph_pyg(data, initial_nodes, terminating_nodes)
    print("Graph visualization complete.")
