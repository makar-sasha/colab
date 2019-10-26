import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_model(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model

def act(state, epsilon, action_size, model):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    act_values = model.predict(state)
    return np.argmax(act_values[0]) 

def replay(batch_size, memory, model, epsilon, epsilon_min, epsilon_decay, gamma):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma *
                        np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

EPISODES = 1000
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
memory = deque(maxlen=2000)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
model = build_model(state_size, action_size, learning_rate)
done = False
batch_size = 32

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        #env.render()
        action = act(state, epsilon, action_size, model)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, EPISODES, time, epsilon))
            break
        if len(memory) > batch_size:
            replay(batch_size, memory, model, epsilon, epsilon_min, epsilon_decay, gamma)