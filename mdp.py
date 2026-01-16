from antlr4 import StdinStream
import markov


def main() -> None:
    chain = markov.parse_mdp(StdinStream())
    chain.normalize_transitions()
    chain.display()
    print("\nSimulated Trace:")
    ex_policy = {"S1": "b", "S2": "b"}  # Example policy for MDP
    trace, reward = chain.walk(start_state=list(chain.states.keys())[0], steps=10, policy=ex_policy)
    print(" -> ".join(trace))
    if reward is not None:
        print(f"Total Reward: {reward}")
    else:
        print("No rewards defined.")


if __name__ == '__main__':
    main()
