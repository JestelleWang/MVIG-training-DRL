# Temporal-Difference Learning

Temporal-Difference (TD) learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas.

## TD Prediction

**Tabular TD(0) for estimating v_π**

```
Input: the policy π to be evaluated
Initialize V(s) arbitrarily
Repeat (for each episode):
	Initialize S
	Repeat (for each step of episode):
		A <- action given by π for S
		Take action A, observe R, S'
		V(S) <- V(S) + α [R + V(S') - γ V(S)]
		S <- S'
	until S is terminal
```

## Advantages of TD Prediction Methods

It is convenient to learn one guess from the next without waiting for an actual outcome, and we still can guarantee convergence to the correct answer.

## Optimality of TD(0)

**batch updating** : Suppose there is  available only a finite amount of experience, say 10 episodes or 100 time steps. In this case, a common approach with incremental learning methods is  to present the experience repeatedly until the method converges upon an answer. Given an approximate value function, V, the increments are computed for every time step t at which a nonterminal state is visited, but the function is changed only once, by the sum of all the increments. Then all the available experience is processed again with the new value function to produce a new overall increment until the value function converges.

## Sarsa: On-Policy TD Control

> Turn to the use of TD prediction methods for the control problem.

First, learn an action-value function rather than a state-value function. Next consider transitions from state-action pair to state-action pair and learn the values of state-action pairs.

**Sarsa: An on-policy TD control algorithm**

```
Initialize Q(s, a), ∀ s ∈ 𝑆, a ∈ A(s), arbitrarily, and Q(terminal-state,﹒)=0
Repeat (for each eposode):
	Initialize S
	Choose A from S using policy derived from Q (e.g., ℇ-greedy)
	Repeat (for each step of episode):
		Take action A, observe R, S'
		Choose A' from S' using policy derived from Q (e.g., ℇ-greedy)
		Q(S, A) <- Q(S, A) + 𝛼[R + 𝛾Q(S', A') - Q(S, A)]
		S <- S'; A <- A';
	until S is terminal
```
## Q-learning: Off-Policy TD Control

One of the early breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as Q-learning (Watkins, 1989), defined by

```
Q(S(t),A(t)) <- Q(S(t), A(t)) + 𝛼[R(t+1) + 𝛾maxQ(S(t+1),a) - Q(S(t),A(t))]
```

**Q-learning: An off-policy TD control algorithm**

```
Initialize Q(s,a), ∀ s ∈ 𝑆, a ∈ A(s), arbitrarily, and Q(terminal-state,﹒)=0
Repeat (for each episode):
	Initialize S
	Repeat (for each step of episode):
		Choose A from S using policy derived from Q (e.g., ℇ-greedy)
		Take action A, observe R, S'
		Q(S,A) <- Q(S,A) + 𝛼[R + 𝛾 maxQ(S', a) - Q(S, A)]
		S <- S'
	until S is terminal
```

## Expected Sarsa




