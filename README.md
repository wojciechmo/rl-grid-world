# Grid world

Compare Q-Learning and Expected Value SARSA.

Goal is to find the best policy of taking actions in grid world in order to receive the biggest possible reward. Agent does not have any prior knowledge about environment. Assumption of deterministic environment, stochasticity is hidden inside agent -> with probability epsilon take a random action, otherwise the best policy action. Penalize each step with small negative reward.

Action value function Q(s,a) is found iteratively with temoporal difference using moving average with one sample (environment is deterministic): Q(s,a) = alpha * (r(s,a) + gamma * v(s')) + (1 - alpha) * Q(s,a).

In Q-Learning algorithm value function v(s) is estimated as maximum of Q(state, action) over possible actions.
In Expected Value SARSA algorithm value function V(s) is estimated as expected value of Q(state, action) over possible actions assuming epsilon-greedy policy.

Different approaches lead to different optimal paths. While Q-Learning (see below on the left side) will prefer the fastest and often more dangerous (do not mind how close to negative rewards we move as long as there is optimistic scenario of happy ending) way to recieve biggest reward, Expected Value SARSA (see below on the right side) will propose safer (avoid negative rewards, move as far from them as possible) path as the best one.

<img src="https://github.com/WojciechMormul/rl-grid-world/blob/master/imgs/1.png" width="300">
<img src="https://github.com/WojciechMormul/rl-grid-world/blob/master/imgs/4.png" width="500">
