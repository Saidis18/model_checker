from antlr4 import StdinStream
import markov


def main() -> None:
    chain = markov.parse_mdp(StdinStream())
    chain.normalize_transitions()
    chain.display()


if __name__ == '__main__':
    main()
