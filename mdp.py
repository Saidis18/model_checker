from antlr4 import StdinStream
import markov


def main() -> None:
    mdp = markov.parse_mdp(StdinStream())
    mdp.normalize_transitions()
    mdp.display()

    print("\nSimulated Trace:")
    ex_policy = {"S1": "b", "S2": "b"}  # Example policy for MDP
    trace = mdp.walk(start_state=list(mdp.states.keys())[0], steps=10, policy=ex_policy)
    print(" -> ".join(f"{state} ({reward})" for state, reward in trace))

    print("\nMarkov Chain from Policy:")
    chain = mdp.markov_chain_from_policy(ex_policy)
    chain.display()
    trace_chain = chain.walk(start_state=list(chain.states.keys())[0], steps=10)
    print(" -> ".join(f"{state} ({reward})" for state, reward in trace_chain))

if __name__ == '__main__':
    main()
