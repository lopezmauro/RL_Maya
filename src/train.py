import os
import imp
import numpy as np
from .enviroment.env import Enviroment
from .ppo import ppo_simple
from .ppo import ppo
from .math_utils import sampling
from .math_utils import vector_math as vm
from maya import cmds
from datetime import datetime
from tensorflow import keras
# episode simulation and nn hyper params
BATCH_SIZE = 1024


def getModelPath():
    test_path = r'D:\dev\RL_Maya\tests'
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
    model_folder = os.path.join(test_path, date_str)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    return model_folder

def createTrainAgents(joints=['joint1', 'joint2', 'joint3'],
                      bundingBox=((-4, -5, -4), (4, 5, 4)),
                      radius=1):
    positions = sampling.poissionDiscSampling(radius=radius, bundingBox=bundingBox, sampleRetrial=5)
    start = 0
    end = 1
    totalTime = 1
    segments_pos = [np.array(cmds.xform(a, q=1, ws=1, t=1)) for a in joints]
    maxY = (segments_pos[-1][1] - segments_pos[0][1])
    locators = list()
    for pos in positions:
        closes_pnt, closerSeg = vm.getCloserSegment(pos, segments_pos)
        zAxis = vm.normalize(pos - closes_pnt)
        yAxis = vm.normalize(segments_pos[closerSeg[1]]-segments_pos[closerSeg[0]])
        xAxis = vm.normalize(np.cross(yAxis, zAxis)) 
        zAxis = vm.normalize(np.cross(xAxis, yAxis))
        loc = cmds.spaceLocator()[0]
        locators.append(loc)
        grp = cmds.group(loc)
        matrix = list(xAxis)
        matrix.append(0)
        matrix.extend(yAxis)
        matrix.append(0)
        matrix.extend(zAxis)
        matrix.append(0)
        matrix.extend(pos)
        matrix.append(1)
        cmds.xform(grp, os=1, m=matrix)
        t = ((pos[1] + segments_pos[0][1]) + maxY)/maxY
        topW = vm.easeInOutCubic(t, start, end, totalTime)
        lowW = 1.0 - topW
        cmds.parentConstraint(joints[0], joints[1], grp, mo=1)
        cmds.parentConstraint(joints[0], grp, e=True, w=lowW)
        cmds.parentConstraint(joints[1], grp, e=True, w=topW)
    return locators


def train(mesh, drivers, agents, n_trains=8, n_episodes=16, epochs=32, batchMax=50, maxFrame=100, convergence=.98):
    model_folder = getModelPath()
    backup_folder = os.path.join(model_folder, 'backup')
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    FILE_NAME = "_".join(drivers)
    rwdFile = os.path.join(model_folder, '{}_epRwd.txt'.format(FILE_NAME))
    score_file = os.path.join(model_folder, '{}_rwd.txt').format(FILE_NAME)
    with open(score_file, 'w') as fd:
        fd.write("")
    agnt_file = os.path.join(model_folder, '{}_agnt_rew.txt').format(FILE_NAME)
    with open(agnt_file, 'w') as fd:
        fd.write("")
    env = Enviroment(agents[0], drivers, mesh, maxFrame)
    ppoAgent = ppo_simple.Agent(env, rwdFile, rwdDiscount=.1)
    rew_history = list()
    for tr_n in range(n_trains):
        # get all agents data --------------
        np.random.shuffle(agents)
        agnt_to_train = agents[:batchMax]
        all_states = np.array([]).reshape(0, ppoAgent.num_states).astype(np.float32)
        all_actions = np.array([]).reshape(0, ppoAgent.num_actions).astype(np.float32)
        all_real_rwds = all_rwds = np.array([]).reshape(0, 1).astype(np.float32)
        for curr_agent in agnt_to_train:
            cmds.hide(agents)
            cmds.showHidden(curr_agent)
            # re initi with random agent
            env.reInit(curr_agent)
            env.reset()
            states, rwds, actions, real_rewd = ppoAgent.get_batch(n_episodes)
            all_states = np.vstack([all_states, states])
            all_actions = np.vstack([all_actions, actions])
            all_rwds = np.vstack([all_rwds, rwds])
            all_real_rwds = np.vstack([all_real_rwds, real_rewd])
        # Train the model ------------------
        # get instance of early stopping callback, it stop if the model doesnt learn
        early_stop_patient = keras.callbacks.EarlyStopping(patience=8)
        # train value nn to approximate rewards for given states
        ppoAgent.critic.fit(all_states, all_rwds, validation_split=0.1,
                            verbose=2, callbacks=[early_stop_patient],
                            epochs=epochs, batch_size=BATCH_SIZE)
        # get the approximate value for all states
        all_values = ppoAgent.critic.predict(all_states)
        # evaluate how advantageous an action is
        all_advg = np.maximum(0, all_rwds - all_values)
        all_advg /= np.max(all_advg)
        all_advg_dim = all_advg.copy()
        for a in range(env.action_space-1):
            all_advg_dim = np.append(all_advg_dim, all_advg, axis=1)
        # train policy net
        ppoAgent.actor.fit([all_states, all_advg_dim], all_actions,
                            verbose=2, epochs=epochs, batch_size=BATCH_SIZE)

        rewd_mean = np.mean(real_rewd)
        rew_history.append(rewd_mean)
        with open(score_file, 'a') as fd:
            fd.write(f"{rewd_mean}\n")
        with open(agnt_file, 'a') as fd:
            fd.write(f"{rewd_mean} : {curr_agent}\n")
        ppoAgent.result_model.save(os.path.join(model_folder, f'{FILE_NAME}_{tr_n:02d}.h5'))
        ppoAgent.actor.save(os.path.join(backup_folder, f'{FILE_NAME}_actor_{tr_n:02d}.h5'))
        ppoAgent.critic.save(os.path.join(backup_folder, f'{FILE_NAME}_critic_{tr_n:02d}.h5'))
        if len(rew_history) > 5 and (np.array(rew_history[-5:]) >= convergence).all():
            print(np.array(rew_history[-5:]))
            print(f"Convergence of {convergence} Reached at train {tr_n}!")
            return



def train_ppo(drivers, agents, n_trains=8, n_episodes=16, epochs=32, batchMax=50, maxFrame=100):
    # Exploration settings
    model_folder = getModelPath()
    backup_folder = os.path.join(model_folder, 'backup')
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    FILE_NAME = "_".join(drivers)
    rwdFile = os.path.join(model_folder, f'{FILE_NAME}_epRwd.txt')
    score_file = os.path.join(model_folder, f'{FILE_NAME}_rwd.txt')
    with open(score_file, 'w') as fd:
        fd.write("")
    agnt_file = os.path.join(model_folder, f'{FILE_NAME}_agnt_rew.txt')
    with open(agnt_file, 'w') as fd:
        fd.write("")
    env = Enviroment(agents[0], drivers, maxFrame)
    ppoAgent = ppo.Agent(env, rwdFile)
    for i in range(n_trains):
        np.random.shuffle(agents)
        agnt_to_train = agents[:batchMax]
        rwd_h = list()
        for curr_agent in agnt_to_train:
            cmds.hide(agents)
            cmds.showHidden(curr_agent)
            # re initi with random agent
            env.reInit(curr_agent)
            env.reset()
            obs, action, pred, reward = ppoAgent.get_batch(n_episodes)
            old_prediction = pred
            pred_values = ppoAgent.critic.predict(obs)
            advantage = reward - pred_values
            all_advg_dim = advantage.copy()
            for a in range(env.action_space-1):
                all_advg_dim = np.append(all_advg_dim, advantage, axis=1)
            ppoAgent.actor.fit([obs, all_advg_dim, old_prediction], [action], batch_size=BATCH_SIZE, epochs=epochs, verbose=2)
            ppoAgent.critic.fit([obs], [reward], batch_size=BATCH_SIZE, epochs=epochs, verbose=2)
            rewd_mean = np.mean(reward)
            with open(score_file, 'a') as fd:
                fd.write(f"{rewd_mean}\n")
            with open(agnt_file, 'a') as fd:
                fd.write(f"{rewd_mean} : {curr_agent}\n")
        score_file = os.path.join(model_folder, '{}_rwd.csv').format(FILE_NAME)
        ppoAgent.result_model.save(os.path.join(model_folder, f'{FILE_NAME}_{i:02d}.h5'))
        ppoAgent.actor.save(os.path.join(backup_folder, f'{FILE_NAME}_actor_{i:02d}.h5'))
        ppoAgent.critic.save(os.path.join(backup_folder, f'{FILE_NAME}_critic_{i:02d}.h5'))
        np.savetxt(score_file, rwd_h)