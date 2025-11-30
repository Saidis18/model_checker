from antlr4 import StdinStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from markov import MarkovChain


class gramPrintListener(gramListener):
    """Parses MDP grammar and populates MarkovChain."""

    def __init__(self) -> None:
        self.chain = MarkovChain()
        
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
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    listener = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    return listener.chain


def main() -> None:
    chain = parse_mdp()
    chain.display()


if __name__ == '__main__':
    main()
