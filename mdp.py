from antlr4 import StdinStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser

        
class gramPrintListener(gramListener):

    def __init__(self):
        pass
        
    def enterStatesrew(self, ctx: gramParser.StatesrewContext):
        ids = [str(x) for x in ctx.ID()]
        rew = [int(str(x)) for x in ctx.INT()]
        print("States: %s" % str([ids[i] + " with reward " + str(rew[i]) for i in range(len(ids))]))
        
    def enterStatesnorew(self, ctx: gramParser.StatesnorewContext):
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx: gramParser.DefactionsContext):
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx: gramParser.TransactContext):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        
    def enterTransnoact(self, ctx: gramParser.TransnoactContext):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

            

def main():
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)

if __name__ == '__main__':
    main()
