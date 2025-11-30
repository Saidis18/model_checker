from antlr4 import StdinStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from typing import List, Dict, Tuple


class MarkovChain:
    """Data structure to hold Markov chain information."""

    def __init__(self) -> None:
        self.states: Dict[str, int] = {}  # state -> reward
        self.actions: List[str] = []
        self.transitions: List[Tuple[str, str, str, int]] = []  # (from, to, action, weight)


class gramPrintListener(gramListener):
    """Parses MDP grammar and populates MarkovChain."""

    def __init__(self, chain: MarkovChain) -> None:
        self.chain = chain
        
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
        self.chain.actions = ids
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
            self.chain.transitions.append((dep, target, "tau", weights[i]))
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))



def parse_mdp() -> MarkovChain:
    """Parse MDP from stdin and return MarkovChain."""
    chain = MarkovChain()
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    listener = gramPrintListener(chain)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    return chain


def display_chain(chain: MarkovChain) -> None:
    """Display the Markov chain using visualizer if available."""
    try:
        from visualizer import MarkovChainVisualizer
        viz = MarkovChainVisualizer()
        for state, reward in chain.states.items():
            viz.add_state(state, reward)
        for action in chain.actions:
            viz.add_action(action)
        for from_state, to_state, action, weight in chain.transitions:
            viz.add_transition(from_state, to_state, action, weight)
        viz.display()
    except ImportError:
        print("\n[Info] Visualizer module not found, skipping visualization")


def main() -> None:
    chain = parse_mdp()
    display_chain(chain)


if __name__ == '__main__':
    main()
