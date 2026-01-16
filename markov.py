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
        self.normalized = False
        self.states: List[str] = []  # list of state names
        self.rewards: Dict[str, int] = {}  # state -> reward mapping
        self.actions: List[str] = [self.no_action_symbol]
        self.transitions: List[MarkovModel._trans_t] = []

    def add_state(self, name: str, reward: int = 0) -> None:
        """Add a state with optional reward."""
        if name not in self.states:
            self.states.append(name)
        self.rewards[name] = reward

    def add_action(self, name: str) -> None:
        """Add an action."""
        if name not in self.actions:
            self.actions.append(name)

    def add_transition(
        self, from_state: str, to_state: str, action: str, weight: int | float
    ) -> None:
        """Add a transition from one state to another."""
        self.transitions.append((from_state, to_state, action, weight))

    @property
    def rewardless(self) -> bool:
        """Return True if the model has rewards, else False."""
        all_reward =  all(self.rewards.get(state, -1) != -1 for state in self.states)
        no_reward = all(self.rewards.get(state, -1) == -1 for state in self.states)
        assert no_reward or all_reward, "Cannot mix rewardless and rewarded models."
        return no_reward
    
    def assert_valid(self) -> None:
        """Assert that the Markov model is valid."""
        no_action_transitions = [t[0] for t in self.transitions if t[2] == self.no_action_symbol]
        if not self.rewardless:
            assert all(self.rewards.get(state, -1) >= 0 for state in self.states), "Rewards must be non-negative."
        for from_state, to_state, action, weight in self.transitions:
            assert from_state in self.states, f"State '{from_state}' not defined."
            assert to_state in self.states, f"State '{to_state}' not defined."
            assert action in self.actions, f"Action '{action}' not defined."
            assert weight >= 0, "Transition weight must be non-negative."
            if action != self.no_action_symbol:
                assert from_state not in no_action_transitions, "Cannot mix actions with no-action transitions."

    def normalize_transitions(self) -> None:
        """Normalize transition weights to probabilities."""
        if self.normalized:
            return  # Already normalized
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
    
    def walk(self, start_state: str, steps: int, policy: None | Dict[str, str] = None) -> List[Tuple[str, int]]:
        """Simulate a walk through the Markov model."""
        assert self.normalized, "Transitions must be normalized before walking."
        assert start_state in self.states, f"Start state '{start_state}' not defined."
        assert self.kind == "MDP" and policy is not None or self.kind == "MC", "Policy must be provided for MDPs."
        if policy is None:
            policy = {}

        current_state = start_state
        current_reward = 0
        path = [(current_state, current_reward)]
        no_reward = self.rewardless

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
            if not no_reward:
                current_reward += self.rewards[current_state] # accumulate reward once we are sure we can move to next state
            current_state = random.choices(next_states, weights=probabilities, k=1)[0]
            path.append((current_state, current_reward))

        return path
    
    def markov_chain_from_policy(self, policy: Dict[str, str]) -> 'MarkovModel':
        """Generate a Markov Chain from the MDP using the given policy."""
        assert self.kind == "MDP", "Can only generate Markov Chain from MDP."
        self.assert_valid()
        mc = MarkovModel()
        mc.states = self.states.copy()
        mc.rewards = self.rewards.copy()
        mc.actions = [self.no_action_symbol]
        
        for from_state, to_state, action, weight in self.transitions:
            if action == policy.get(from_state, self.no_action_symbol):
                mc.add_transition(from_state, to_state, self.no_action_symbol, weight)
        
        mc.assert_valid()
        mc.normalize_transitions()
        return mc
    
    def _MC_iter_accessibility(self, start_state: str, end_state: str, steps: int) -> float:
        """Compute the probability of reaching end_state from start_state in at least n steps."""
        assert self.kind == "MC", "Accessibility can only be computed for Markov Chains."
        self.assert_valid()
        self.normalize_transitions()
        assert start_state in self.states, f"Start state '{start_state}' not defined."
        assert end_state in self.states, f"End state '{end_state}' not defined."

        # Initialize probability distribution: 1.0 at start_state, 0 elsewhere
        prob_dist: Dict[str, float] = {state: 0.0 for state in self.states}
        prob_dist[start_state] = 1.0

        # Iteratively apply transition probabilities for 'steps' iterations
        # Treat end_state as absorbing (once reached, stays there)
        for _ in range(steps):
            next_dist: Dict[str, float] = {state: 0.0 for state in self.states}
            
            for from_state, prob in prob_dist.items():
                if prob > 0:
                    if from_state == end_state:
                        # End state is absorbing - probability stays there
                        next_dist[end_state] += prob
                    else:
                        # Get all transitions from this state
                        outgoing = [t for t in self.transitions if t[0] == from_state]
                        for _, to_state, _, transition_prob in outgoing:
                            next_dist[to_state] += prob * transition_prob
            
            prob_dist = next_dist

        return prob_dist[end_state]
    
    def iter_accessibility(self, start_state: str, end_state: str, policy: Dict[str, str], steps: int) -> float:
        """Compute the accessibility probability from start_state to end_state in given steps."""
        self.assert_valid()
        if self.kind == "MC":
            return self._MC_iter_accessibility(start_state, end_state, steps)
        else:
            mc = self.markov_chain_from_policy(policy)
            return mc._MC_iter_accessibility(start_state, end_state, steps)
        
    def _MC_expected_reward(self, start_state: str, end_state: str, steps: int) -> float:
        """Compute the expected reward to reach end_state from start_state in given steps."""
        assert self.kind == "MC", "Expected reward can only be computed for Markov Chains."   
        self.normalize_transitions()
        assert start_state in self.states, f"Start state '{start_state}' not defined."
        assert end_state in self.states, f"End state '{end_state}' not defined."

        # Initialize probability and reward distributions
        prob_dist: Dict[str, float] = {state: 0.0 for state in self.states}
        reward_dist: Dict[str, float] = {state: 0.0 for state in self.states}
        prob_dist[start_state] = 1.0

        for _ in range(steps):
            next_prob_dist: Dict[str, float] = {state: 0.0 for state in self.states}
            next_reward_dist: Dict[str, float] = {state: 0.0 for state in self.states}

            for from_state, prob in prob_dist.items():
                if prob > 0:
                    if from_state == end_state:
                        next_prob_dist[end_state] += prob
                        next_reward_dist[end_state] += reward_dist[end_state] + prob * self.rewards[end_state]
                    else:
                        outgoing = [t for t in self.transitions if t[0] == from_state]
                        for _, to_state, _, transition_prob in outgoing:
                            next_prob_dist[to_state] += prob * transition_prob
                            next_reward_dist[to_state] += (reward_dist[from_state] + prob * self.rewards[from_state]) * transition_prob

            prob_dist = next_prob_dist
            reward_dist = next_reward_dist

        if prob_dist[end_state] > 0:
            expected_reward = reward_dist[end_state] / prob_dist[end_state]
        else:
            expected_reward = 0.0

        return expected_reward
    
    def expected_reward(self, start_state: str, end_state: str, policy: Dict[str, str], steps: int) -> float:
        """Compute the expected reward to reach end_state from start_state in given steps."""
        self.assert_valid()
        if self.kind == "MC":
            return self._MC_expected_reward(start_state, end_state, steps)
        else:
            mc = self.markov_chain_from_policy(policy)
            return mc._MC_expected_reward(start_state, end_state, steps)

    def display(self, output_name: str) -> None:
        """Display the Markov model using Graphviz."""

        graph = Digraph(output_name, format="png")
        graph.attr(rankdir="LR")
        graph.attr("node", shape="circle", style="filled", fillcolor="lightblue")
        graph.attr("edge", color="gray")

        # Add states as nodes
        for state in self.states:
            reward = self.rewards.get(state, 0)
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
            graph.render(output_name, view=False, cleanup=True)
            print(f"[Visualizer] Graph saved to {output_name}.png")
        except Exception as e:
            print(f"[Visualizer] Error rendering graph: {e}")
            self._print_summary()

    def _print_summary(self) -> None:
        """Print a text summary of the Markov model."""
        print("\n=== Markov Model Summary ===")
        print(f"States: {self.states}")
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
            self.chain.add_state(ids[i], rew[i])
        print("States: %s" % str([ids[i] + " with reward " + str(rew[i]) for i in range(len(ids))]))
        
    def enterStatesnorew(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        for state in ids:
            self.chain.add_state(state, -1)
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
