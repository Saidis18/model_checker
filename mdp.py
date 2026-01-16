from antlr4 import StdinStream
import markov


def main() -> None:
    mdp = markov.parse_mdp(StdinStream())
    mdp.normalize_transitions()
    mdp.display("mdp_model")

    print("\nSimulated Trace:")
    ex_policy = {"S0": "a", "S1": "b", "S2": "a", "S3": "a", "S4": "b"}  # Example policy for MDP
    trace = mdp.walk(start_state=mdp.states[0], steps=10, policy=ex_policy)
    print(" -> ".join(f"{state} ({reward})" for state, reward in trace))

    print("\nMarkov Chain from Policy:")
    chain = mdp.markov_chain_from_policy(ex_policy)
    chain.display("markov_chain_from_policy")
    trace_chain = chain.walk(start_state=chain.states[0], steps=10)
    print(" -> ".join(f"{state} ({reward})" for state, reward in trace_chain))

    print("\nAccessibility Probability:")
    start, end, steps = "S0", "S3", 5
    prob = mdp.iter_accessibility(start, end, steps, policy=ex_policy) # type: ignore
    print(f"Probability of reaching {end} from {start} in {steps} steps: {prob:.4f}")

    print("\nExpected Reward:")
    expected_reward = mdp.expected_reward(start, end, steps, policy=ex_policy) # type: ignore
    print(f"Expected reward to reach {end} from {start} in {steps} steps: {expected_reward:.4f}")

    print("\nTransition Matrix of MDP under Policy:")
    matrix = mdp.get_matrix_representation(policy=ex_policy) # type: ignore
    print(matrix)

    print("\nAccessibility probability via linear system:")
    prob_linear = mdp.accessibility(start, end, policy=ex_policy) # type: ignore
    print(f"Probability of reaching {end} from {start} via linear system: {prob_linear:.4f}")

if __name__ == '__main__':
    main()
