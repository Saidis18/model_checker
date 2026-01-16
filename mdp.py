from antlr4 import StdinStream
import markov


def main() -> None:
    chain = markov.parse_mdp(StdinStream())
    chain.normalize_transitions()
    chain.display()
    print("\nSimulated Trace:")
    ex_policy = {"S1": "b", "S2": "b"}  # Example policy for MDP
    trace = chain.walk(start_state=list(chain.states.keys())[0], steps=10, policy=ex_policy)
    print(" -> ".join(trace))


if __name__ == '__main__':
    main()
