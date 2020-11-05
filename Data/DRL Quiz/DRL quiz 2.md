

## Finite Markov Decision Process

Dynamics: 

$p\left(s^{\prime}, r \mid s, a\right) \doteq \operatorname{Pr}\left\{S_{t}=s^{\prime}, R_{t}=r \mid S_{t-1}=s, A_{t-1}=a\right\}$ for all $s^{\prime}, s \in \mathcal{S}, r \in \mathcal{R},$ and $a \in \mathcal{A}(s)$

$\sum_{s^{\prime} \in \mathcal{S}} \sum_{r \in \mathcal{R}} p\left(s^{\prime}, r \mid s, a\right)=1,$ for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$

state-transition probabilities:

 $p\left(s^{\prime} \mid s, a\right) \doteq \operatorname{Pr}\left\{S_{t}=s^{\prime} \mid S_{t-1}=s, A_{t-1}=a\right\}=\sum_{r \in \mathcal{R}} p\left(s^{\prime}, r \mid s, a\right)$

expected rewards for state-action pairs: 

$r(s, a) \doteq \mathbb{E}\left[R_{t} \mid S_{t-1}=s, A_{t-1}=a\right]=\sum_{r \in \mathcal{R}} r \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime}, r \mid s, a\right)$

 state–action–next-state: 

$r\left(s, a, s^{\prime}\right) \doteq \mathbb{E}\left[R_{t} \mid S_{t-1}=s, A_{t-1}=a, S_{t}=s^{\prime}\right]=\sum_{r \in \mathcal{R}} r \frac{p\left(s^{\prime}, r \mid s, a\right)}{p\left(s^{\prime} \mid s, a\right)}$

Return: Given reward is 1, return =:

 $G_{t}=\sum_{k=0}^{\infty} \gamma^{k}=\frac{1}{1-\gamma}$

$\begin{aligned} G_{t} & \doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\gamma^{3} R_{t+4}+\cdots \\ &=R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\gamma^{2} R_{t+4}+\cdots\right) \\ &=R_{t+1}+\gamma G_{t+1} \end{aligned}$

Value Function of a state under policy:

$v_{\pi}(s) \doteq \mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right]=\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \mid S_{t}=s\right],$ for all $s \in \mathcal{S}$

$\begin{aligned} v_{\pi}(s) & \doteq \mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] \\ &=\sum_{a} \pi(a \mid s) \sum_{s^{\prime}} \sum_{r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t+1}=s^{\prime}\right]\right] \\ &=\sum_{a} \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right], \quad \text { for all } s \in \mathcal{S} \text {(Bellman equation for value function)} \end{aligned}$ 

Action value function for policy $\pi$

$q_{\pi}(s, a) \doteq \mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s, A_{t}=a\right]=\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \mid S_{t}=s, A_{t}=a\right]$

Optimal value function (Bellman Optimality equation):

Intuitively, the Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state

$\begin{aligned} v_{*}(s) &=\max _{a \in \mathcal{A}(s)} q_{\pi_{*}}(s, a) \\ &=\max _{a} \mathbb{E}_{\pi_{*}}\left[G_{t} \mid S_{t}=s, A_{t}=a\right] \\ &=\max _{a} \mathbb{E}_{\pi_{*}}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s, A_{t}=a\right] \\ &=\max _{a} \mathbb{E}\left[R_{t+1}+\gamma v_{*}\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=a\right] \\ &=\max _{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{*}\left(s^{\prime}\right)\right] \end{aligned}$

Optimal action state function:

$\begin{aligned} q_{*}(s, a) &=\mathbb{E}\left[R_{t+1}+\gamma \max _{a^{\prime}} q_{*}\left(S_{t+1}, a^{\prime}\right) \mid S_{t}=s, A_{t}=a\right] \\ &=\sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \max _{a^{\prime}} q_{*}\left(s^{\prime}, a^{\prime}\right)\right] \end{aligned}$

![image-20201103222625193](DRL quiz 2/image-20201103222625193.png)

For finite MDPs, the Bellman optimality equation for $v_\pi$ has a unique solution.

For each state s , there will be one or more actions at which the maximum is obtained in the Bellman optimality equation. Any policy that assigns nonzero probability only to these actions is an optimal policy (?)

The undiscounted formulation is appropriate for episodic tasks, in which the agent–environment interaction breaks naturally into episodes; the discounted formulation is appropriate for continuing tasks, in which the interaction does not naturally break into episodes but continues without limit.

## Monte Carlo Methods

To ensure that well-defined returns are available, here we define Monte Carlo methods only for episodic tasks. That is, we assume experience is divided into episodes, and that all episodes eventually terminate no matter what actions are selected. Only on the completion of an episode are value estimates and policies changed.

### GPI: Solves the Monte Carlo Problem

We use the term generalized policy iteration (GPI) to refer to the general idea of letting policy-evaluation and policy-improvement processes interact, independent of the granularity and other details of the two processes



### First-visit MC prediction, for estimating $V \approx v_{\pi}$

Input: a policy $\pi$ to be evaluated Initialize:
$V(s) \in \mathbb{R},$ arbitrarily, for all $s \in \mathcal{S}$ Returns $(s) \leftarrow$ an empty list, for all $s \in \mathcal{S}$
Loop forever (for each episode):
Generate an episode following $\pi: S_{0}, A_{0}, R_{1}, S_{1}, A_{1}, R_{2}, \ldots, S_{T-1}, A_{T-1}, R_{T}$
$G \leftarrow 0$
Loop for each step of episode, $t=T-1, T-2, \ldots, 0$
$$
G \leftarrow \gamma G+R_{t+1}
$$
Unless $S_{t}$ appears in $S_{0}, S_{1}, \ldots, S_{t-1}$ :

​															Append $G$ to Returns$\left(S_{t}\right)$
$$
V\left(S_{t}\right) \leftarrow \text { average }\left(\text {Returns}\left(S_{t}\right)\right)
$$
Both first-visit MC and every-visit MC converge to v ⇡ ( s ) as the number of visits (or first visits) to s goes to infinity

Each average is itself an unbiased estimate, and the standard deviation of its error falls as 1 / p n , where n is the number of returns averaged.

One way to do this is by specifying that the episodes start in a state–action pair, and that every pair has a nonzero probability of being selected as the start. This guarantees that all state–action pairs will be visited an infinite number of times in the limit of an infinite number of episodes. We call this the assumption of exploring starts.



Backup Diagram for Monte Carlo control(?)



### Monte Carlo Control

#### policy improvement theorem

$\begin{aligned} q_{\pi_{k}}\left(s, \pi_{k+1}(s)\right) &=q_{\pi_{k}}\left(s, \underset{a}{\arg \max } q_{\pi_{k}}(s, a)\right) \\ &=\max _{a} q_{\pi_{k}}(s, a) \\ & \geq q_{\pi_{k}}\left(s, \pi_{k}(s)\right) \\ & \geq v_{\pi_{k}}(s) \end{aligned}$

![image-20201103234902181](DRL quiz 2/image-20201103234902181.png)

![image-20201103234838812](DRL quiz 2/image-20201103234838812.png)

That any  $\epsilon$-greedy policy with respect to q ⇡ is an improvement over any $\epsilon$-soft policy $\pi$ is assured by the policy improvement theorem

Relative probability:

 $\rho_{t: T-1} \doteq \frac{\prod_{k=t}^{T-1} \pi\left(A_{k} \mid S_{k}\right) p\left(S_{k+1} \mid S_{k}, A_{k}\right)}{\prod_{k=t}^{T-1} b\left(A_{k} \mid S_{k}\right) p\left(S_{k+1} \mid S_{k}, A_{k}\right)}=\prod_{k=t}^{T-1} \frac{\pi\left(A_{k} \mid S_{k}\right)}{b\left(A_{k} \mid S_{k}\right)}$

The importance sampling ratio ends up depending only on the two policies and the sequence, not on the MDP.

![image-20201104102517406](DRL quiz 2/image-20201104102517406.png)

#### weighted importance sampling:

$V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t: T(t)-1} G_{t}}{\sum_{t \in \mathcal{T}(s)} \rho_{t: T(t)-1}}$

The every-visit methods for ordinary and weighed importance sampling are both biased, though, again, the bias falls asymptotically to zero as the number of samples increases.

$V_{n} \doteq \frac{\sum_{k=1}^{n-1} W_{k} G_{k}}{\sum_{k=1}^{n-1} W_{k}}, \quad n \geq 2$

##### Incremental update

$V_{n+1} \doteq V_{n}+\frac{W_{n}}{C_{n}}\left[G_{n}-V_{n}\right], \quad n \geq 1$
$C_{n+1} \doteq C_{n}+W_{n+1}$ where $C_{0} \doteq 0$

#### Off-policy MC prediction (policy evaluation) for estimating $Q \approx q_{\pi}$

![image-20201104104243956](DRL quiz 2/image-20201104104243956.png)

#### Off-Policy Monte Carlo control: based on GPI and weighted importance sampling

![image-20201104104256970](DRL quiz 2/image-20201104104256970.png)

### Monte Carlo Tree Search

1. Selection. Starting at the root node, a tree policy based on the action values attached to the edges of the tree traverses the tree to select a leaf node.
2. Expansion. On some iterations (depending on details of the application), the tree is expanded from the selected leaf node by adding one or more child nodes reached from the selected node via unexplored actions.
3. Simulation. From the selected node, or from one of its newly-added child nodes (if any), simulation of a complete episode is run with actions selected by the rollout policy. The result is a Monte Carlo trial with actions selected first by the tree policy and beyond the tree by the rollout policy.
4. Backup. The return generated by the simulated episode is backed up to update, or to initialize, the action values attached to the edges of the tree traversed by the tree policy in this iteration of MCTS. No values are saved for the states and actions visited by the rollout policy beyond the tree. Figure 8.10 illustrates this by showing a backup from the terminal state of the simulated trajectory directly to the state-action node in the tree where the rollout policy began (though in general, the entire return over the simulated trajectory is backed up to this state-action node).

## Temporal-Di↵erence Learning

### Tabular $\mathrm{TD}(0)$ for estimating $v_{\pi}$

![image-20201104104720688](DRL quiz 2/image-20201104104720688.png)

### Bootstrapping

$\begin{aligned} v_{\pi}(s) & \doteq \mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] \end{aligned}$
$=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s\right]$

![image-20201104104854506](DRL quiz 2/image-20201104104854506.png)

### TD target

![截屏2020-11-04 17.57.50](/Users/leon/Library/Mobile Documents/com~apple~CloudDocs/Courses/博文/MarkdownImages/Data/算法分析/DRL quiz 2/截屏2020-11-04 17.57.50.png)

### TD Error

$\delta_{t} \doteq R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)$

#### Monte carlo error: Sum of TD Error

$G_{t}-V\left(S_{t}\right)=R_{t+1}+\gamma G_{t+1}-V\left(S_{t}\right)+\gamma V\left(S_{t+1}\right)-\gamma V\left(S_{t+1}\right) \quad$ 
$=\delta_{t}+\gamma\left(G_{t+1}-V\left(S_{t+1}\right)\right)$
$=\delta_{t}+\gamma \delta_{t+1}+\gamma^{2}\left(G_{t+2}-V\left(S_{t+2}\right)\right)$
$=\delta_{t}+\gamma \delta_{t+1}+\gamma^{2} \delta_{t+2}+\cdots+\gamma^{T-t-1} \delta_{T-1}+\gamma^{T-t}\left(G_{T}-V\left(S_{T}\right)\right)$
$=\delta_{t}+\gamma \delta_{t+1}+\gamma^{2} \delta_{t+2}+\cdots+\gamma^{T-t-1} \delta_{T-1}+\gamma^{T-t}(0-0)$
$=\sum_{k=t}^{T-1} \gamma^{k-t} \delta_{k}$

This identity is not exact if V is updated during the episode (as it is in TD(0)), but if the step size is small then it may still hold approximately

### Advantage

Obviously, TD methods have an advantage over DP methods in that they do not require a model of the environment, of its reward and next-state probability distributions.

For any fixed policy $\pi$ , TD(0) has been proved to converge to $v_\pi$ , in the mean for a constant step-size parameter if it is sufficiently small, and with probability 1 if the step-size parameter decreases according to
the usual stochastic approximation conditions (2.7).

### Optimality

Under batch updating, $\mathrm{TD}(0)$ converges deterministically to a single answer independent of the step-size parameter, $\alpha,$ as long as $\alpha$ is chosen to be sufficiently small. The constant$\alpha$ MC method also converges deterministically under the same conditions, but to a different answer. 

Example 6.4 illustrates a general difference between the estimates found by batch $\mathrm{TD}(0)$ and batch Monte Carlo methods. Batch Monte Carlo methods always find the estimates that minimize mean-squared error on the training set, whereas batch TD(0) always finds the estimates that would be exactly correct for the maximum-likelihood model of the Markov process. In general, the maximum-likelihood estimate of a parameter is the parameter value whose probability of generating the data is greatest. In this case, the maximum-likelihood estimate is the model of the Markov process formed in the obvious way from the observed episodes: the estimated transition probability from $i$ to $j$ is the fraction of observed transitions from $i$ that went to $j,$ and the associated expected reward is the average of the rewards observed on those transitions. Given this model, we can compute the estimate of the value function that would be exactly correct if the model were exactly correct. This is called the certainty-equivalence estimate because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated. In general, batch TD(0) converges to the certainty-equivalence estimate.

### Sarsa: On-policy TD

Sarsa (on-policy TD control) for estimating $Q \approx q_{*}$

![image-20201104113412596](DRL quiz 2/image-20201104113412596.png)

### Q-learning: Off policy TD

Q-learning (off-policy TD control) for estimating $\pi \approx \pi_{*}$

![image-20201104113436521](DRL quiz 2/image-20201104113436521.png)

### Expected reward

$\begin{aligned} Q\left(S_{t}, A_{t}\right) & \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma \mathbb{E}_{\pi}\left[Q\left(S_{t+1}, A_{t+1}\right) \mid S_{t+1}\right]-Q\left(S_{t}, A_{t}\right)\right] \\ & \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma \sum_{a} \pi\left(a \mid S_{t+1}\right) Q\left(S_{t+1}, a\right)-Q\left(S_{t}, A_{t}\right)\right] \end{aligned}$

###  maximization bias

As we discussed above, there will be a positive maximization bias if we use the maximum of the estimates as an estimate of the maximum of the true values. One way to view the problem is that it is due to using the same samples (plays) both to determine the maximizing action and to estimate its value. Suppose we divided the plays in two sets and used them to learn two independent estimates, call them $Q_{1}(a)$ and $Q_{2}(a),$ each an estimate of the true value $q(a),$ for all $a \in \mathcal{A}$. We could then use one estimate, say $Q_{1},$ to determine the maximizing action $A^{*}=\arg \max _{a} Q_{1}(a),$ and the other, $Q_{2},$ to provide the estimate of its value, $Q_{2}\left(A^{*}\right)=Q_{2}\left(\arg \max _{a} Q_{1}(a)\right) .$ This estimate will then be unbiased in the sense that $\mathbb{E}\left[Q_{2}\left(A^{*}\right)\right]=q\left(A^{*}\right) .$ We can also repeat the process with the role of the two estimates reversed to yield a second unbiased estimate $Q_{1}\left(\arg \max _{a} Q_{2}(a)\right) .$ This is the idea of double learning. Note that although we learn two estimates, only one estimate is updated on each play; double learning doubles the memory requirements, but does not increase the amount of computation per step.

![image-20201104135756946](DRL quiz 2/image-20201104135756946.png)

## n-step Bootstrapping

### n-step TD Prediction

#### n-step return:

$G_{t: t+n} \doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1} R_{t+n}+\gamma^{n} V_{t+n-1}\left(S_{t+n}\right)$

#### natural state-value learning algorithm for using n-step returns

$V_{t+n}\left(S_{t}\right) \doteq V_{t+n-1}\left(S_{t}\right)+\alpha\left[G_{t: t+n}-V_{t+n-1}\left(S_{t}\right)\right], \quad 0 \leq t<T$

![image-20201104140151453](DRL quiz 2/image-20201104140151453.png)

![image-20201104140618840](DRL quiz 2/image-20201104140618840.png)

![image-20201104140626866](DRL quiz 2/image-20201104140626866.png)

The off-policy version of $n$ -step Expected Sarsa would use the same update as above for $n$ -step Sarsa except that the importance sampling ratio would have one less factor in
it. That is, the above equation would use $\rho_{t+1: t+n-1}$ instead of $\rho_{t+1: t+n},$ and of course it would use the Expected Sarsa version of the $n$ -step return $(7.7) .$

$G_{t: t+n} \doteq R_{t+1}+\cdots+\gamma^{n-1} R_{t+n}+\gamma^{n} \bar{V}_{t+n-1}\left(S_{t+n}\right), \quad t+n<T$

## DQN/Deep Sarsa

### Loss function:

Let \(\mu(S)\) denote how much time we spend in each state \(s\) under policy \(\pi\) then:
$$
J(w)=\sum_{n=1}^{|\delta|} \mu(S)\left[v_{\pi}(S)-\hat{v}(S, \mathbf{w})\right]^{2} \sum_{s \in \mathcal{S}} \mu(S)=1
$$
In contrast to:
$$
J_{2}(w)=\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}}\left[v_{\pi}(S)-\hat{v}(S, \mathbf{w})\right]^{2}
$$

### On-policy state distribution

Then the un-normalized on-policy state probability satisfies the following recursions:
$$
\begin{array}{c}
\eta(s)=h(s)+\sum_{\bar{s}} \eta(\bar{s}) \sum_{a} \pi(a \mid \bar{s}) p(s \mid \bar{s}, a), \forall s \in \delta \\
\mu(s)=\frac{\eta(s)}{\sum_{s^{\prime}} \eta\left(s^{\prime}\right)}, \forall s \in \delta
\end{array}
$$

### Update Rule

$$\begin{aligned} \Delta \mathbf{w} &=-\frac{1}{2} \alpha \nabla_{\mathbf{w}} J(\mathbf{w}) \\ &=\alpha \mathbb{E}_{\pi}\left[\left(v_{\pi}(S)-\hat{v}(S, \mathbf{w})\right) \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w})\right] \end{aligned}$$
$$
w_{n+1}=w_{n}-\frac{1}{2} \alpha \nabla_{w} J\left(w_{n}\right)
$$

## Policy Gradient Methods

These methods seek to maximize performance, so their updates approximate gradient ascent in $J:$
$$
\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_{t}+\alpha \widehat{\nabla J\left(\boldsymbol{\theta}_{t}\right)}
$$
where $\widehat{\nabla J\left(\boldsymbol{\theta}_{t}\right)} \in \mathbb{R}^{d^{\prime}}$ is a stochastic estimate whose expectation approximates the gradient of the performance measure with respect to its argument $\theta_{t}$. All methods that follow this general schema we call policy gradient methods, whether or not they also learn an approximate value function.

### Advantage:

One advantage of parameterizing policies according to the soft-max in action preferences is that the approximate policy can approach a deterministic policy, whereas with $\varepsilon$ -greedy action selection over action values there is always an $\varepsilon$ probability of selecting a random action.

A second advantage of parameterizing policies according to the soft-max in action preferences is that it enables the selection of actions with arbitrary probabilities.

### Theorem

$\nabla J(\boldsymbol{\theta}) \propto \sum_{s} \mu(s) \sum_{a} q_{\pi}(s, a) \nabla \pi(a \mid s, \boldsymbol{\theta}),$

### REINFORCE: Monte Carlo Policy Gradient

$\begin{aligned} \nabla J(\boldsymbol{\theta}) & \propto \sum_{s} \mu(s) \sum_{a} q_{\pi}(s, a) \nabla \pi(a \mid s, \boldsymbol{\theta}) \\ &=\mathbb{E}_{\pi}\left[\sum_{a} q_{\pi}\left(S_{t}, a\right) \nabla \pi\left(a \mid S_{t}, \boldsymbol{\theta}\right)\right] \end{aligned}$

$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_{t}+\alpha \sum_{a} \hat{q}\left(S_{t}, a, \mathbf{w}\right) \nabla \pi\left(a \mid S_{t}, \boldsymbol{\theta}\right)$ where $\hat{q}$ is some learned approximation to $q_{\pi}$.

#### Update：

$\begin{aligned} \nabla J(\boldsymbol{\theta}) &=\mathbb{E}_{\pi}\left[\sum_{a} \pi\left(a \mid S_{t}, \boldsymbol{\theta}\right) q_{\pi}\left(S_{t}, a\right) \frac{\nabla \pi\left(a \mid S_{t}, \boldsymbol{\theta}\right)}{\pi\left(a \mid S_{t}, \boldsymbol{\theta}\right)}\right] \\ &=\mathbb{E}_{\pi}\left[q_{\pi}\left(S_{t}, A_{t}\right) \frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}\right)}\right] \\ &=\mathbb{E}_{\pi}\left[G_{t} \frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}\right)}\right], \quad \text { (replacing } a \text { by the sample } \left.A_{t} \sim \pi\right) \end{aligned}$

So $\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_{t}+\alpha G_{t} \frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}$

#### Code：

![image-20201104143930716](DRL quiz 2/image-20201104143930716.png)

##### Difference

1. compact expression $\nabla \ln \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)$ for the fractional vector $\frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}$ in $(13.8) .$ That
   these two expressions for the vector are equivalent follows from the identity $\nabla \ln x=\frac{\nabla x}{x}$.
2. The second difference between the pseudocode update and the REINFORCE update equation (13.8) is that the former includes a factor of $\gamma^{t} .$ This is because, as mentioned earlier, in the text we are treating the non-discounted case $(\gamma=1)$ while in the boxed algorithms we are giving the algorithms for the general discounted case. 
3. $\nabla \ln \pi(a \mid s, \boldsymbol{\theta})=\mathbf{x}(s, a)-\sum_{b} \pi(b \mid s, \boldsymbol{\theta}) \mathbf{x}(s, b)$

### REINFORCE with Baseline

arbitrary baseline $b(s)$:

$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_{\boldsymbol{a}}\left(q_{\pi}(s, a)-b(s)\right) \nabla \pi(a \mid s, \boldsymbol{\theta})$

$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_{t}+\alpha\left(G_{t}-b\left(S_{t}\right)\right) \frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}$

![image-20201104144449341](DRL quiz 2/image-20201104144449341.png)

### Actor–Critic Methods

Although the REINFORCE-with-baseline method learns both a policy and a state-value
function, we do not consider it to be an actor–critic method because its state-value function
is used only as a baseline, not as a critic. That is, it is not used for bootstrapping (updating
the value estimate for a state from the estimated values of subsequent states), but only
as a baseline for the state whose estimate is being updated. （Bootstrapping in RL can be read as "using one or more estimated values in the update step *for the same kind* of estimated value".）

####  One-step actor–critic methods replace the full return of REINFORCE (13.11) with the one-step return

$\begin{aligned} \boldsymbol{\theta}_{t+1} & \doteq \boldsymbol{\theta}_{t}+\alpha\left(G_{t: t+1}-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)} \\ &=\boldsymbol{\theta}_{t}+\alpha\left(R_{t+1}+\gamma \hat{v}\left(S_{t+1}, \mathbf{w}\right)-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)} \\ &=\boldsymbol{\theta}_{t}+\alpha \delta_{t} \frac{\nabla \pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)}{\pi\left(A_{t} \mid S_{t}, \boldsymbol{\theta}_{t}\right)} \end{aligned}$

![image-20201104144608575](DRL quiz 2/image-20201104144608575.png)

The generalizations to the forward view of $n$ -step methods and then to a $\lambda$ -return algorithm are straightforward. The one-step return in (13.12) is merely replaced by $G_{t: t+n}$ or $G_{t}^{\lambda}$ respectively. The backward view of the $\lambda$ -return algorithm is also straightforward, using separate eligibility traces for the actor and critic, each after the patterns in Chapter 12. Pseudocode for the complete algorithm is given in the box below.

![image-20201104144615909](DRL quiz 2/image-20201104144615909.png)

### Policy Gradient for Continuing Problems 



![image-20201104144708616](DRL quiz 2/image-20201104144708616.png)





