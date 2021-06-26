# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
#diabliling to not compute the compleate graph since the code 
# is form TF1 and is not optimized for TF2 TODO:optimize for TF2
disable_eager_execution() 


LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
NOISE = 2.0 # Exploration noise
GAMMA = 0.1
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))
        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss

class Agent:
    def __init__(self, env, rwdFile, num_layers=2, hidden_size=512,
                 epsilon=1.0, minEpsilon=.1):
        self.env = env
        self.rwdFile = rwdFile
        self.epsilon = epsilon
        self.minEpsilon = minEpsilon
        self.num_actions = self.env.action_space
        self.num_states = self.env.observation_space
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.critic = self.build_critic()
        self.actor, self.result_model = self.build_actor()
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward_over_time = []
        self.name = "PPO_test"
        #self.writer = SummaryWriter(self.name)
        self.gradient_steps = 0
        #self.upper_bound = env.action_space.high[0]
        #self.lower_bound = env.action_space.low[0]
        with open(self.rwdFile,'w') as fd:
            fd.write("")

    def build_actor(self):
        state_input = Input(shape=(self.num_states,))
        advantage = Input(shape=(self.num_actions,))
        old_prediction = Input(shape=(self.num_actions,))

        x = Dense(self.hidden_size, activation='tanh')(state_input)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_size, activation='tanh')(x)

        out_actions = Dense(self.num_actions, name='output', activation='tanh')(x)

        model_predict = Model(inputs=[state_input], outputs=[out_actions])
        model_train = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model_train.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model_train.summary()

        return model_train, model_predict

    def build_critic(self):

        state_input = Input(shape=(self.num_states,))
        x = Dense(self.hidden_size, activation='tanh')(state_input)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_size, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self, observation, noise):
        #p = self.actor.predict([self.observation.reshape(1, self.num_states), self.dummy_action, self.dummy_value])
        p = self.result_model.predict(observation.reshape(1, self.num_states))[0]
        action = p + np.random.normal(loc=0, scale=noise, size=p[0].shape)
        return action, p

    def transform_reward(self, rewards):
        #if self.val is True:
        #    self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        #else:
        #    self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * GAMMA
        return rewards

    def get_batch(self, n_episodes):
        # batch = [[], [], [], []]
        batch_obs = list()
        batch_action = list()
        batch_pred = list()
        batch_reward = list()
        epsilonGradient = np.linspace(self.minEpsilon, self.epsilon, n_episodes)[::-1]
        for j in range(n_episodes):
            # tmp_batch = [[], [], [], []]
            tmp_obs = list()
            tmp_action = list()
            tmp_pred = list()
            tmp_real_rwrd = list()
            temp_reward = list()
            observation = self.env.reset()
            done = False
            while not done:
                action, predicted_action = self.get_action(observation, epsilonGradient[j])
                _, rel_reward, _, _ = self.env.step(predicted_action, addFrame=False)
                observation_, reward, done, info = self.env.step(action)
                temp_reward.append(reward)
                tmp_obs.append(self.observation)
                tmp_action.append(action)
                tmp_pred.append(predicted_action)
                tmp_real_rwrd.append(rel_reward)
                observation = observation_

            #rewards = self.discount_rewards()
            reward = self.transform_reward(temp_reward)
            batch_obs.extend(tmp_obs)
            batch_action.extend(tmp_action)
            batch_pred.extend(tmp_pred)
            batch_reward.extend(reward)
            print(f'Episode avg. reward {np.mean(tmp_real_rwrd)}')
            with open(self.rwdFile, 'a') as fd:
                fd.write("{}\n".format(np.mean(tmp_real_rwrd)))
            self.reset_env()
        obs = np.array(batch_obs)
        action = np.array(batch_action)
        pred = np.array(batch_pred)
        reward = np.reshape(np.array(batch_reward), (len(batch_reward), 1))
        #pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

