from typing import Dict, List, Tuple
import os
import subprocess
from graphviz import Digraph


class MarkovChain:
    """Visualizer for Markov chains using Graphviz."""

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
        """Display the Markov chain using Graphviz."""

        graph = Digraph("markov_chain", format="png")
        graph.attr(rankdir="LR")
        graph.attr("node", shape="circle", style="filled", fillcolor="lightblue")
        graph.attr("edge", color="gray")

        # Add states as nodes
        for state, reward in self.states.items():
            label = state + (f"\n(r={reward})" if reward != 0 else "")
            graph.node(state, label=label)

        # Group transitions by edge to combine labels
        edge_transitions: Dict[Tuple[str, str], List[str]] = {}
        for from_state, to_state, action, weight in self.transitions:
            key = (from_state, to_state)
            if key not in edge_transitions:
                edge_transitions[key] = []
            edge_transitions[key].append(f"{action}({weight})")

        # Add edges with combined labels
        for (from_state, to_state), labels in edge_transitions.items():
            label = "\\n".join(labels)
            graph.edge(from_state, to_state, label=label)

        # Render and display
        try:
            output_path = "markov_chain"
            graph.render(output_path, view=False, cleanup=True)
            print(f"\n[Visualizer] Graph saved to {output_path}.png")
        except Exception as e:
            print(f"\n[Visualizer] Error rendering graph: {e}")
            self._print_summary()

    def _print_summary(self) -> None:
        """Print a text summary of the Markov chain."""
        print("\n=== Markov Chain Summary ===")
        print(f"States: {list(self.states.keys())}")
        print(f"Actions: {self.actions}")
        print("Transitions:")
        for from_state, to_state, action, weight in self.transitions:
            print(f"  {from_state} --[{action}:w={weight}]--> {to_state}")
