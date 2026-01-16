from antlr4 import StdinStream
import markov


def main() -> None:
    chain = markov.parse_mdp(StdinStream())
    chain.display()


if __name__ == '__main__':
    main()
