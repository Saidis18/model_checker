from typing import Dict, List, Tuple

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_GRAPH_LIBS = True
except ImportError:
    HAS_GRAPH_LIBS = False


class MarkovChainVisualizer:
    """Simple visualizer for Markov chains."""

    def __init__(self) -> None:
        self.states: Dict[str, int] = {}  # state -> reward
        self.actions: List[str] = []
        self.transitions: List[Tuple[str, str, str, int]] = []  # (from, to, action, weight)

    def add_state(self, name: str, reward: int = 0) -> None:
        """Add a state with optional reward."""
        self.states[name] = reward

    def add_action(self, name: str) -> None:
        """Add an action."""
        if name not in self.actions:
            self.actions.append(name)

    def add_transition(
        self, from_state: str, to_state: str, action: str, weight: int
    ) -> None:
        """Add a transition from one state to another."""
        self.transitions.append((from_state, to_state, action, weight))

    def display(self) -> None:
        """Display the Markov chain as a graph."""
        if not HAS_GRAPH_LIBS:
            print("\n[Visualizer] networkx/matplotlib not available. Install with:")
            print("  pip install networkx matplotlib")
            self._print_summary()
            return

        G = nx.DiGraph()

        # Add nodes
        for state, reward in self.states.items():
            label = f"{state}" + (f"\n(r={reward})" if reward != 0 else "")
            G.add_node(state, label=label)

        # Add edges with labels
        edge_labels = {}
        for from_state, to_state, action, weight in self.transitions:
            G.add_edge(from_state, to_state)
            key = (from_state, to_state)
            if key not in edge_labels:
                edge_labels[key] = []
            edge_labels[key].append(f"{action}({weight})")

        # Combine multiple transitions on same edge
        edge_labels_final = {k: ", ".join(v) for k, v in edge_labels.items()}

        # Draw
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42, k=2)
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1500)
        nx.draw_networkx_labels(
            G, pos, {n: G.nodes[n]["label"] for n in G.nodes()}
        )
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->"
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels_final, font_size=8)

        plt.title("Markov Chain")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _print_summary(self) -> None:
        """Print a text summary of the Markov chain."""
        print("\n=== Markov Chain Summary ===")
        print(f"States: {list(self.states.keys())}")
        print(f"Actions: {self.actions}")
        print("Transitions:")
        for from_state, to_state, action, weight in self.transitions:
            print(f"  {from_state} --[{action}:w={weight}]--> {to_state}")
