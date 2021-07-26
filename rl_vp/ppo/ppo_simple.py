
import logging
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, env, rwdFile="", epsilon=1.0, minEpsilon=.1,
                 rwdDiscount=.1, hidden_size=512, num_layers=1):
        self.env = env
        self.rwdFile = rwdFile
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = self.env.action_space
        self.num_states = self.env.observation_space
        self.observation = self.env.reset()
        self.name = "PPO_test"
        self.epsilon = epsilon
        self.minEpsilon = minEpsilon
        self.rwdDiscount = rwdDiscount
        self.critic = self.build_critic()
        self.actor, self.result_model = self.build_actor()
        if self.rwdFile:
            with open(self.rwdFile, 'w') as fd:
                fd.write("")

    def build_critic(self):
        # nn topology
        in_layer = layers.Input(shape=[self.num_states], name="input_x")
        x = layers.Dense(self.hidden_size, activation='tanh')(in_layer)
        for _ in range(self.num_layers - 1):
            x = layers.Dense(self.hidden_size, activation='tanh')(x)
        out_layer = layers.Dense(1, activation="linear", name="out")(x)

        model = Model(inputs=in_layer, outputs=out_layer)
        model.compile(loss='mse', optimizer='RMSProp')

        return model

    def build_actor(self):
        # nn topology
        in_layer = layers.Input(shape=[self.num_states], name="input_x")
        x = layers.Dense(self.hidden_size, activation='tanh')(in_layer)
        for _ in range(self.num_layers - 1):
            x = layers.Dense(self.hidden_size, activation='tanh')(x)
        out_layer = layers.Dense(self.num_actions,
                                 activation="linear",
                                 name="out")(x)

        # advg_ph is a placeholder for inputing rewards value into
        # the model, making them avaiable for the loss functions
        advg_ph = layers.Input(shape=[self.num_actions], name="rewards")

        # custom mean squared error loss with rewards
        def custom_loss(y_true, y_pred):
            #return advg_ph*K.square(y_pred - y_true)
            return K.mean(advg_ph*K.square(y_pred - y_true), axis=-1)

        # create two models that share layers
        # one has rewards as inputs (for training only)
        model_predict = Model(inputs=[in_layer], outputs=out_layer)
        model_train = Model(inputs=[in_layer, advg_ph], outputs=out_layer)
        model_train.compile(loss=custom_loss, optimizer='RMSProp')
        model_predict.compile(loss='mse', optimizer='RMSProp')
        return model_train, model_predict

    def load_models(self, actor_file, critic_file, predict_file):
        self.actor = load_model(actor_file)
        self.critic = load_model(critic_file)
        self.result_model = load_model(predict_file)


    def discount_rewards(self, r, discount):
        # adapted from Martin Gorner's Github project:
        # https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd    
        r = np.array(r)
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * discount + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def get_batch(self, n_episodes, train_number=0):
        all_states = np.array([]).reshape(0, self.num_states).astype(np.float32)
        all_actions = np.array([]).reshape(0, self.num_actions).astype(np.float32)
        all_dsc_real_rwds = all_dsc_rwds = np.array([]).reshape(0, 1).astype(np.float32)
        # Decay epsilon gradient for noise reduction each step
        epsilonGradient = np.linspace(self.minEpsilon, self.epsilon, n_episodes)[::-1]
        for j in range(n_episodes):
            # re/start simulation
            for attr, values in self.env.animations:
                done = False
                ep_rwds = np.array([]).reshape(0, 1).astype(np.float32)
                ep_real_rwds = np.array([]).reshape(0, 1).astype(np.float32)
                self.env.currAttr = attr
                self.env.currAnim = values
                observation = self.env.reset().astype(np.float32)
                while not done:
                    action = self.result_model.predict(observation.reshape(1, self.num_states))
                    _, real_reward, _, _ = self.env.step(action[0], addFrame=False)
                    action = np.random.normal(action, epsilonGradient[j])  # randomize action
                    observation_, reward, done, info = self.env.step(action[0])
                    # store all values
                    all_states = np.vstack([all_states, observation])
                    all_actions = np.vstack([all_actions, action])
                    ep_real_rwds = np.vstack([ep_real_rwds, real_reward])
                    ep_rwds = np.vstack([ep_rwds, reward])
                    observation = observation_
                if self.rwdFile:
                    with open(self.rwdFile,'a') as fd:
                        fd.write("{}\n".format(np.mean(ep_rwds)))
                dsc_rwds = self.discount_rewards(ep_rwds, self.rwdDiscount)
                all_dsc_rwds = np.vstack([all_dsc_rwds, dsc_rwds])
                real_dsc_rwds = self.discount_rewards(ep_real_rwds, self.rwdDiscount)
                all_dsc_real_rwds = np.vstack([all_dsc_real_rwds, real_dsc_rwds])
            logger.info(f'{self.env.agent} Train {train_number} Episode {j}  Avg. reward {np.mean(all_dsc_real_rwds)}')
        return all_states, all_dsc_rwds, all_actions, all_dsc_real_rwds
