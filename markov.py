from typing import Dict, List, Tuple
from graphviz import Digraph
from antlr4 import StdinStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser


class MarkovModel:
    """Represents a Markov Chain or Markov Decision Process."""
    def __init__(self) -> None:
        self.no_action_symbol = "*"
        self.states: Dict[str, int] = {}  # state -> reward
        self.actions: List[str] = [self.no_action_symbol]
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
    
    def assert_valid(self) -> None:
        """Assert that the Markov model is valid."""
        for from_state, to_state, action, weight in self.transitions:
            assert from_state in self.states, f"State '{from_state}' not defined."
            assert to_state in self.states, f"State '{to_state}' not defined."
            assert action in self.actions, f"Action '{action}' not defined."
            assert weight >= 0, "Transition weight must be non-negative."

    def display(self) -> None:
        """Display the Markov model using Graphviz."""

        graph = Digraph("markov_model", format="png")
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
            edge_transitions[key].append(f"{action} -- {weight}")

        # Add edges with combined labels
        for (from_state, to_state), labels in edge_transitions.items():
            label = "\\n".join(labels)
            graph.edge(from_state, to_state, label=label)

        # Render and display
        try:
            output_path = "markov_model"
            graph.render(output_path, view=False, cleanup=True)
            print(f"\n[Visualizer] Graph saved to {output_path}.png")
        except Exception as e:
            print(f"\n[Visualizer] Error rendering graph: {e}")
            self._print_summary()

    def _print_summary(self) -> None:
        """Print a text summary of the Markov model."""
        print("\n=== Markov Model Summary ===")
        print(f"States: {list(self.states.keys())}")
        print(f"Actions: {self.actions}")
        print("Transitions:")
        for from_state, to_state, action, weight in self.transitions:
            print(f"  {from_state} --[{action}:w={weight}]--> {to_state}")


class gramPrintListener(gramListener):
    """Parses Markov model grammar and populates MarkovModel."""

    def __init__(self) -> None:
        self.chain = MarkovModel()
        
    def enterStatesrew(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        rew = [int(str(x)) for x in ctx.INT()]
        for i in range(len(ids)):
            self.chain.states[ids[i]] = rew[i]
        print("States: %s" % str([ids[i] + " with reward " + str(rew[i]) for i in range(len(ids))]))
        
    def enterStatesnorew(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        for state in ids:
            self.chain.states[state] = 0
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        self.chain.actions.extend(ids)
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        for i, target in enumerate(ids):
            self.chain.transitions.append((dep, target, act, weights[i]))
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        for i, target in enumerate(ids):
            self.chain.transitions.append((dep, target, self.chain.no_action_symbol, weights[i]))
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))


def parse_mdp(in_stream: StdinStream) -> MarkovModel:
    """Parse MDP from stdin and return MarkovChain."""
    lexer = gramLexer(in_stream)
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    listener = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    listener.chain.assert_valid()
    return listener.chain
