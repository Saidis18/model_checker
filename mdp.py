from antlr4 import StdinStream
import markov


def main() -> None:
    mdp = markov.parse_mdp(StdinStream())
    mdp.normalize_transitions()
    mdp.display("mdp_model")

    print("\nSimulated Trace:")
    ex_policy = {"S1": "a", "S2": "b"}  # Example policy for MDP
    trace = mdp.walk(start_state=mdp.states[0], steps=10, policy=ex_policy)
    print(" -> ".join(f"{state} ({reward})" for state, reward in trace))

    print("\nMarkov Chain from Policy:")
    chain = mdp.markov_chain_from_policy(ex_policy)
    chain.display("markov_chain_from_policy")
    trace_chain = chain.walk(start_state=chain.states[0], steps=10)
    print(" -> ".join(f"{state} ({reward})" for state, reward in trace_chain))

    print("\nAccessibility Probability:")
    start, end, steps = "S0", "S2", 100
    prob = mdp.iter_accessibility(start, end, ex_policy, steps)
    print(f"Probability of reaching {end} from {start} in {steps} steps: {prob:.4f}")

if __name__ == '__main__':
    main()
