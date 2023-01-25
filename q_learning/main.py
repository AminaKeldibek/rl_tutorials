class Evaluator:
    __sum_of_rewards_per_episode = [0]

    def log_sum_of_rewards_per_episode(self, reward, episode):
        if (episode + 1) > len(self.sum_of_rewards_per_episode):
            self.sum_of_rewards_per_episode.append(0)
        self.sum_of_rewards_per_episode[episode] += reward

    @property
    def sum_of_rewards_per_episode(self):
        return self.__sum_of_rewards_per_episode


class QLearning:
    '''Implements Off-policy control with Q Learning.'''

    def __init__(self, env, num_states, num_actions, alpha, epsilon, gamma):
        '''Parameters
        ----------
        env:         gym.core.Environment, open gym environment object
        num_states:  integer, number of states in the environment
        num_actions: integer, number of possible actions
        alpha:       float, step size, (0, 1]
        epsilon:     float, the epsilon parameter used for exploration
        gamma:       float, discount factor, small > 0
        '''
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = np.zeros((self.num_states, self.num_actions))

        self.env.reset()

    def run_q_learning(self, num_episodes, evaluator, verbose=True):
        '''Runs Q learning

        Parameters
        ----------
        num_episodes: integer, number of episodes to run to train RL agent

        Returns
        ----------
        self.policy:         list of integers of length self.num_states, final policy
        '''
        terminated = False
        for i in range(num_episodes):
            self.env.reset()
            state = np.random.choice(self.num_states, 1)[0]
            terminated = False
            while not terminated:
                action, next_state, reward, terminated = self.generate_next_step(state)
                # evaluator.log_sum_of_rewards_per_episode(reward, i)
                if not isinstance(next_state, int):
                    raise TypeError('Integer is expected as a next state')
                self.evaluate_policy(state, action, reward, next_state)

                state = next_state

        # Once training is finished, calculate and return policy using argmax approach
        final_policy = np.argmax(self.Q, axis=1)

        return final_policy

    def generate_next_step(self, state):
        '''Generates action in state and outcome (next_state, reward, terminated)of taking this action.

        Parameters
        ----------
        state: int, current state of the agent

        Returns
        ----------
        action:      int, action that agent takes in state
        observation: object, observation as a result of agent taking action in state, format is specific to the environment
        reward:      float
        terminated:  bool, whether a terminal state (as defined under the MDP of the task) is reached.
        '''
        random_action = self.env.action_space.sample()
        action = self.get_epsilon_greedy_action(state, random_action)

        observation, reward, terminated, _ = self.env.step(action)

        return (action, observation, reward, terminated)

    def evaluate_policy(self, state, action, reward, next_state):
        '''Updates Q value for a specific state and action pair.

        Parameters
        ----------
        state:      int, current state
        action:     int, action taken in state
        reward:     float, reward as a result of taking action in state
        next_state: int
        '''
        est_reward = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] = self.Q[state][action] + self.alpha * (est_reward - self.Q[state][action])

    def argmax(self, state: int) -> int:
        """
        Finds and returns greedy action.

        Parameters
        ----------
        state: int, state for which greedy action should be selected

        Returns
        ----------
        action: int, corresponds to the index of the greedy action

        """
        return int(np.argmax(self.Q[state]))

    def get_epsilon_greedy_action(self, state, random_action):
        '''Returns next action using epsilon greedy approach.

        Parameters
        ----------
        state:         int, current state
        random_action: int, action sampled from current state

        Returns
        ----------
        next_action: int, either greedy or random action
        '''
        prob = np.random.random()

        if prob < (1 - self.epsilon):
            return self.argmax(state)

        return random_action