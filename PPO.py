import itertools
import random
import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import InputLayer, Dense, BatchNormalization
import matplotlib.pyplot as plt
from copy import deepcopy
batch_size = 100
reward_list = []

SMOOTH_NUM = 100
def smooth(l):
    if len(l) < SMOOTH_NUM:
        return l
    tmp = []
    current_sum = 0
    for i in range(len(l)):
        current = l[i]
        current_sum += current
        tmp.append(current_sum/(i+1))
        if i == SMOOTH_NUM-2:
            break
    for i in range(SMOOTH_NUM-1, len(l)):
        tmp.append(sum(l[i-(SMOOTH_NUM-1):i+1])/SMOOTH_NUM)
    l = tmp
    return l

def plotting(l):
    length = len(l)
    index = list(range(length))
    plt.plot(index, smooth(l))
    plt.savefig('stat.png')

class Normal():
    def __init__(self, m):
        if len(m.shape) == 1:
            self.batch_mode = False
            self.d = m.shape[0]
            self.m = m
            self.s = np.diag([0.3 for _ in range(self.d)])
        elif len(m.shape) == 2:
            self.batch_mode = True
            self.b = m.shape[0]
            self.d = m.shape[1]
            self.m = m
            self.s = np.diag([0.3 for _ in range(self.d)])

    def prob(self, x):
        a = np.exp((-1/2)*np.sum((x-self.m)**2))
        b = np.sqrt((2*np.pi)**self.d)
        return a/b

    def prob_train(self, x):
        if self.batch_mode:
            a = tf.cast(tf.exp((-1 / 2) * tf.reduce_sum((x - self.m) ** 2, axis=1)), tf.float64)
            b = tf.cast(tf.sqrt((2 * np.pi) ** self.d), tf.float64)
            return a / b
        else:
            a = tf.cast(tf.exp((-1 / 2) * tf.reduce_sum((x - self.m) ** 2)), tf.float64)
            b = tf.cast(tf.sqrt((2 * np.pi) ** self.d), tf.float64)
            return a / b

    def sample(self):
        return np.random.multivariate_normal(self.m, self.s)

class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []

    def append(self, memory_sample):
        #0: prev_s, 1: action, 2: r, 3: dist.prob(action)
        self.states.append(memory_sample[0])
        self.actions.append(memory_sample[1])
        self.rewards.append(memory_sample[2])
        self.action_probs.append(memory_sample[3])

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []

class Replay():
    def __init__(self):
        self.max_len = 10000
        self.states = []
        self.actions = []
        self.action_probs = []
        self.gae = []
        self.oracle_values = []

    def append_gae(self, l):
        if len(l) + len(self.gae) > self.max_len:
            self.gae = self.gae[len(l):]
            self.gae += l
        else:
            self.gae += l
        return

    def append_oracle(self, l):
        if len(l) + len(self.oracle_values) > self.max_len:
            self.oracle_values = self.oracle_values[len(l):]
            self.oracle_values += l
        else:
            self.oracle_values += l
        return

    def append_states(self, l):
        if len(l) + len(self.states) > self.max_len:
            self.states = self.states[len(l):]
            self.states += l
        else:
            self.states += l
        return

    def append_actions(self, l):
        if len(l) + len(self.actions) > self.max_len:
            self.actions = self.actions[len(l):]
            self.actions += l
        else:
            self.actions += l
        return

    def append_action_probs(self, l):
        if len(l) + len(self.action_probs) > self.max_len:
            self.action_probs = self.action_probs[len(l):]
            self.action_probs += l
        else:
            self.action_probs += l
        return

    def refresh(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.gae = []
        self.oracle_values = []
        return


def calculate_value(memory, model):
    states = np.array(memory.states)
    values = np.array(model.value(states))
    values = values.reshape((-1))
    return values

def calculate_delta(memory, values, gamma, done):
    delta = []
    rewards = memory.rewards
    length = len(values)
    for i in range(length):
        if i < length -1:
            delta.append(rewards[i] + gamma*values[i+1] - values[i])
        else:
            if done:
                delta.append(rewards[i] - values[i])
    return delta

def calculate_gae(delta, r):
    length = len(delta)
    gae = []
    running = 0
    for i in reversed(range(length)):
        running = delta[i] + r*running
        gae.append(running)
    gae.reverse()
    return gae

def optimize(replay, model, epsilon, k):
    states = replay.states
    actions = replay.actions
    p_olds = replay.action_probs
    gae = replay.gae
    oracle_values = replay.oracle_values
    oracle_values = np.array(oracle_values, dtype=np.float64)
    states = np.array(states, dtype=np.float64)
    actions = np.array(actions, dtype=np.float64)
    p_olds = np.array(p_olds, dtype=np.float64)
    gae = np.array(gae, dtype=np.float64)
    length = gae.shape[0]
    if length > 1:
        m = np.mean(gae)
        s = np.std(gae)
        gae = (gae - m) / s
    else:
        m = np.mean(gae)
        gae = gae - m

    idx_arr = np.arange(length)
    np.random.shuffle(idx_arr)
    sum_actor_loss = 0
    sum_critic_loss = 0
    for i in range(length//batch_size):
        _oracle_values = tf.convert_to_tensor(oracle_values[idx_arr[i*batch_size:(i+1)*batch_size]], dtype=tf.float64)
        _states = tf.convert_to_tensor(states[idx_arr[i*batch_size:(i+1)*batch_size]], dtype=tf.float64)
        _actions = tf.convert_to_tensor(actions[idx_arr[i*batch_size:(i+1)*batch_size]], dtype=tf.float64)
        _p_olds = tf.convert_to_tensor(p_olds[idx_arr[i*batch_size:(i+1)*batch_size]], dtype=tf.float64)
        _gae = tf.convert_to_tensor(gae[idx_arr[i*batch_size:(i+1)*batch_size]], dtype=tf.float64)
        a, c = train(_oracle_values, _states, _actions, _p_olds, _gae, model, i, length)
        if sum_actor_loss == 0:
            sum_actor_loss = a
        else:
            sum_actor_loss = sum_actor_loss + a
        if sum_critic_loss == 0:
            sum_critic_loss = c
        else:
            sum_critic_loss = sum_critic_loss + c
    sum_actor_loss = sum_actor_loss/(length//batch_size)
    sum_critic_loss = sum_critic_loss/(length//batch_size)
    print('actor loss, critic loss: ', sum_actor_loss, sum_critic_loss)




def train(oracle_values, states, actions, p_olds, gae, model, i, length):
    actor_loss = 0
    critic_loss = 0
    with tf.GradientTape() as tape_actor:
        action_means = model.action(states)
        dist = Normal(action_means)
        p_nows = dist.prob_train(actions)
        r = p_nows / p_olds
        a = gae * r
        b = gae * tf.clip_by_value(r, 1 - epsilon, 1 + epsilon)
        tmp = tf.minimum(a, b)
        loss = -tf.reduce_mean(tmp)
        actor_loss = loss
    grads = tape_actor.gradient(loss, model.actor_vars())
    model.opt_actor.apply_gradients(zip(grads, model.actor_vars()))
    with tf.GradientTape() as tape_critic:
        predicted_values = model.value(states)
        predicted_values = tf.reshape(predicted_values, [-1, ])
        loss = tf.reduce_mean((predicted_values - oracle_values) ** 2)
        critic_loss = loss
    grads = tape_critic.gradient(loss, model.critic_vars())
    model.opt_critic.apply_gradients(zip(grads, model.critic_vars()))
    return actor_loss, critic_loss

def calculate_oracle_values(values, gae):
    length = len(gae)
    oracle_values = np.zeros((length))
    for i in range(length):
        oracle_values[i] = values[i] + gae[i]
    return list(oracle_values)

class PPO_Network(Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_input = InputLayer((state_size, ))
        self.action_batch_1 = BatchNormalization()
        self.action_dense_1 = Dense(units=50, activation='elu')
        self.action_batch_2 = BatchNormalization()
        self.action_dense_2 = Dense(units=10, activation='elu')
        self.value_batch_1 = BatchNormalization()
        self.value_dense_1 = Dense(units=50, activation='elu')
        self.value_batch_2 = BatchNormalization()
        self.value_dense_2 = Dense(units=10, activation='elu')
        self.action_out = Dense(units=action_size, activation='linear')
        self.value_out = Dense(units=1)
        self.opt_actor = tf.keras.optimizers.Adam(lr=1e-4)
        self.opt_critic = tf.keras.optimizers.Adam(lr=3e-4)

    def action(self, state):
        tmp = self.state_input(state)
        tmp = self.action_batch_1(tmp)
        tmp = self.action_dense_1(tmp)
        tmp = self.action_dense_2(tmp)
        tmp = self.action_batch_2(tmp)
        action = self.action_out(tmp)
        action = tf.clip_by_value(action, -1, 1)
        return action

    def actor_vars(self):
        return list(itertools.chain(self.action_batch_1.trainable_variables, self.action_dense_1.trainable_variables,
                                    self.action_batch_2.trainable_variables, self.action_dense_2.trainable_variables,
                                    self.action_out.trainable_variables))

    def critic_vars(self):
        return list(itertools.chain(self.value_batch_1.trainable_variables, self.value_dense_1.trainable_variables,
                                    self.value_batch_2.trainable_variables, self.value_dense_2.trainable_variables,
                                    self.value_out.trainable_variables))

    def value(self, state):
        tmp = self.state_input(state)
        tmp = self.value_batch_1(tmp)
        tmp = self.value_dense_1(tmp)
        tmp = self.value_dense_2(tmp)
        tmp = self.value_batch_2(tmp)
        value = self.value_out(tmp)
        return value

def preprocess(obj):
    return np.array(obj, dtype=np.float64)

def sample_random_action(action_size):
    return np.random.uniform(-1, 1, (action_size, ))

if __name__ == "__main__":
    mode = 'train_'
    resume = False
    tf.config.gpu.set_per_process_memory_growth(True)
    if mode == 'train':
        env = gym.make("BipedalWalker-v2")
        replay = Replay()
        gamma = 0.95
        lamb = 0.91
        epsilon = 0.2
        random_action_ratio = 0.01
        epochs = 15
        max_t = 201
        max_step = np.inf
        cooltime = 3000
        cooltime_cnt = 0
        if resume:
            tmp_state = preprocess(np.zeros((24,)))
            model = PPO_Network(24, 4)
            model.value(tmp_state.reshape(1, -1))
            model.action(tmp_state.reshape(1, -1))
            model.load_weights("model.h5")
        else:
            model = PPO_Network(24, 4)
        for e in range(100000):
            t = 0
            step = 0
            total_reward = 0
            print('e:', e)
            d = False
            memory = Memory()
            prev_s = preprocess(env.reset())
            prev_r = 0
            prev_action = None
            action = None
            random_flag = False
            random_cnt = 0
            if e % 100 == 1:
                random_action_ratio *= 0.95
            while not d:
                cooltime_cnt += 1
                t += 1
                step += 1
                action_mean = model.action(prev_s.reshape(1, -1))
                if step%80 == 0:
                    print(action_mean)
                dist = Normal(action_mean[0])
                if random_flag:
                    random_cnt -= 1
                    if random_cnt == 0:
                        random_flag = False
                    else:
                        action = prev_action
                else:
                    tmp = np.random.randint(0, 10000)
                    if tmp > random_action_ratio*10000 :
                        action = dist.sample()
                        prev_action = action
                    else:
                        random_flag = True
                        random_cnt = 5
                        action = sample_random_action(4)
                        prev_action = action
                s, r, d, _ = env.step(action)
                if step > max_step:
                    d = True
                prev_r = r
                s = preprocess(s)
                total_reward += r
                memory_sample = list(prev_s), list(action), r, dist.prob(action)
                memory.append(memory_sample)
                prev_s = s

                if e % 10 == 0:
                    env.render()

                if t == max_t or d:
                    t = 0
                    values = calculate_value(memory, model)
    #                print('values: ', values)
                    delta = calculate_delta(memory, values, gamma, d)
    #                print('delta: ', delta)
                    gae = calculate_gae(delta, gamma*lamb)
                    oracle_values = calculate_oracle_values(values, gae)
                    if len(gae) != len(memory.states):
                        replay.append_states(deepcopy(memory.states[:-1]))
                        replay.append_actions(deepcopy(memory.actions[:-1]))
                        replay.append_action_probs(deepcopy(memory.action_probs[:-1]))
                        replay.append_gae(deepcopy(gae))
                        replay.append_oracle(deepcopy(oracle_values))
                    else:
                        replay.append_states(deepcopy(memory.states))
                        replay.append_actions(deepcopy(memory.actions))
                        replay.append_action_probs(deepcopy(memory.action_probs))
                        replay.append_gae(deepcopy(gae))
                        replay.append_oracle(deepcopy(oracle_values))
#                    if len(replay.gae) > 399:
#                        for k in range(epochs):
#                            optimize(replay, model, epsilon, k)
#                        replay.refresh()
                    memory.reset()
                if cooltime == cooltime_cnt:
                    cooltime_cnt = 0
                    print('In episode ', e)
                    for k in range(epochs):
                        print('epoch: ', k)
                        optimize(replay, model, epsilon, k)
                    replay.refresh()

            print('total_reward: ', total_reward)
            reward_list.append(total_reward)
            if e % 100 == 1 and e > 10000:
                plotting(reward_list)
                model.save_weights("model.h5")
    else:
        env = gym.make("BipedalWalker-v2")
        tmp_state = preprocess(np.zeros((24,)))
        model = PPO_Network(24, 4)
        model.value(tmp_state.reshape(1, -1))
        model.action(tmp_state.reshape(1, -1))
        model.load_weights("model.h5")
        for e in range(10):
            total_reward = 0
            print('e:', e)
            d = False
            prev_s = preprocess(env.reset())
            while not d:
                action_mean = model.action(prev_s.reshape(1, -1))
                action = action_mean[0]
                print(action_mean[0])
                s, r, d, _ = env.step(action)
                s = preprocess(s)
                total_reward += r
                prev_s = s
                env.render()
            print('total_reward: ', total_reward)



