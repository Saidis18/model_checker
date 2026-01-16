from typing import Dict, List, Tuple
from graphviz import Digraph
from antlr4 import StdinStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import random


class MarkovModel:
    """Represents a Markov Chain or Markov Decision Process."""
    _trans_t = Tuple[str, str, str, int] | Tuple[str, str, str, float]  # (from, to, action, weight or probability)
    def __init__(self) -> None:
        self.no_action_symbol = "*"
        self.states: Dict[str, int] = {}  # state -> reward
        self.actions: List[str] = [self.no_action_symbol]
        self.transitions: List[MarkovModel._trans_t] = []

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

    @property
    def rewardless(self) -> bool:
        """Return True if the model has rewards, else False."""
        all_reward =  all(reward != -1 for reward in self.states.values())
        no_reward = all(reward == -1 for reward in self.states.values())
        assert no_reward or all_reward, "Cannot mix rewardless and rewarded models."
        return no_reward
    
    def assert_valid(self) -> None:
        """Assert that the Markov model is valid."""
        no_action_transitions = [t[0] for t in self.transitions if t[2] == self.no_action_symbol]
        if not self.rewardless:
            assert all(reward >= 0 for reward in self.states.values()), "Rewards must be non-negative."
        for from_state, to_state, action, weight in self.transitions:
            assert from_state in self.states, f"State '{from_state}' not defined."
            assert to_state in self.states, f"State '{to_state}' not defined."
            assert action in self.actions, f"Action '{action}' not defined."
            assert weight >= 0, "Transition weight must be non-negative."
            if action != self.no_action_symbol:
                assert from_state not in no_action_transitions, "Cannot mix actions with no-action transitions."

    def normalize_transitions(self) -> None:
        """Normalize transition weights to probabilities."""
        sum_weights: Dict[Tuple[str, str], int | float] = {}
        for from_state, to_state, action, weight in self.transitions:
            sum_weights[(from_state, action)] = sum_weights.get((from_state, action), 0) + weight
        
        normalized_transitions: List[MarkovModel._trans_t] = []
        for from_state, to_state, action, weight in self.transitions:
            total = sum_weights[(from_state, action)]
            if total > 0:
                probability = weight / total
            else:
                probability = 0.0
            normalized_transitions.append((from_state, to_state, action, probability))

        self.transitions = normalized_transitions
        self.normalized = True

    @property
    def kind(self) -> str:
        """Return 'MDP' if actions are used, else 'MC'."""
        return "MDP" if any(t[2] != self.no_action_symbol for t in self.transitions) else "MC"
    
    def walk(self, start_state: str, steps: int, policy: None | Dict[str, str] = None) -> Tuple[List[str], int | None]:
        """Simulate a walk through the Markov model."""
        assert self.normalized, "Transitions must be normalized before walking."
        assert start_state in self.states, f"Start state '{start_state}' not defined."
        assert self.kind == "MDP" and policy is not None or self.kind == "MC", "Policy must be provided for MDPs."
        if policy is None:
            policy = {}

        current_state = start_state
        path = [current_state]
        reward = 0

        for _ in range(steps):
            possible_transitions = [t for t in self.transitions if t[0] == current_state]
            if not possible_transitions:
                break  # No outgoing transitions, end the walk
            action = policy.get(current_state, self.no_action_symbol)
            action_transitions = [t for t in possible_transitions if t[2] == action]
            if not action_transitions:
                break  # No transitions for the chosen action, end the walk
            probabilities = [t[3] for t in action_transitions]
            next_states = [t[1] for t in action_transitions]
            reward += self.states[current_state] # accumulate reward once we are sure we can move to next state
            current_state = random.choices(next_states, weights=probabilities, k=1)[0]
            path.append(current_state)

        if self.rewardless:
            reward = None
        return path, reward

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
            self.chain.states[state] = -1
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
