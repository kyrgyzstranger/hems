import csv
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import sgd

class ExperienceReplay(object):

    # constructor
    def __init__(self, max_memory=100, discount=.99):
        
        self.max_memory = max_memory
        self.memory = []
        self.discount = discount

    # add experience to memory
    def remember(self, experience):
        
        self.memory.append(experience)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # get batch of experiences from memory 
    def get_batch(self, model, num_inputs, batch_size=24):
        
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        targets = np.zeros((inputs.shape[0], num_actions))
        
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0:4]
            end_of_episode = state_t['hour'] == 23
            inp = np.array(list(state_t.values()))
            inp = np.expand_dims(inp, axis=0)
            inputs[i:i+1] = inp
            targets[i] = model.predict(inp)[0]
            Q_sa = np.max(model.predict(inp)[0])
            if end_of_episode:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

def act(state, action, next_state):
    reward = 0.0
    bat_charge = 0.0
    car_charge = 0.0
    # actions involving shift (deferrment) of car charge to one hour
    if action == 1 or action == 3 or action == 5:
        # execute car charge if already shifted to two hours
        # or if it is the last state, i.e. hour of episode, i.e. day
        if state['shift'] == 2 or next_state['hour'] == 23:
            car_charge = state['car']
        # otherwise, shift car charge to the next state
        else:
            next_state['car'] += state['car']
            next_state['shift'] += 1
    # actions involving battery manipulations
    if action in [2, 3, 4, 5]:
        # actions involving battery discharge
        if action in [2, 3]:
            # discharge battery if enough charge level
            bat_charge = -5.0 if state['bat'] >= 5.0 else 0.0
        # actions involving recharge
        else:
            # recharge battery if not full
            bat_charge = 5.0 if state['bat'] <= 10.0 else 0.0
        # alter battery level for the next state
        next_state['bat'] += bat_charge
    # calculate reward
    reward = state['price'] * (state['gen'] - state['reg'] - car_charge - bat_charge)

    return reward, next_state

# load data and initialize environment
def init_env(file_path):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, file_path)
    raw_env = list()
    csv_rfile = open(file_path,)
    csv_reader = csv.reader(csv_rfile, delimiter=',')
    for row in csv_reader:
        raw_env.append(row)
    csv_rfile.close()
    env = []
    day = []
    for i in range(len(raw_env)):
        parsed_time = time.strptime(raw_env[i][0], '%Y-%m-%d %H:%M:%S')
        reg = float(raw_env[i][1])
        car = float(raw_env[i][2])
        gen = float(raw_env[i][3])
        price = float(raw_env[i][4])
        state = {'day_of_week': parsed_time.tm_wday, 'hour': parsed_time.tm_hour, 'reg': reg, 'car': car, 'gen': gen, 'bat': 0.0, 'shift': 0, 'price': price}
        day.append(state)
        if(i + 1)%24 == 0:
            env.append(day)
            day = []
    return env

# configure neural network
def get_model(num_inputs, num_actions, hidden_size):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=num_inputs, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.00001), loss="mse")
    return model

def train(model, env, epochs, verbose):
    loss_hist = []
    epsilon = .1
    for e in range(epochs):
        train_loss = 0.0
        # choose a random episode, i.e. day and assign an initial battery level randomly chosen
        # from values 0, 5, 10, 15 kWh
        index = np.random.randint(0, len(env)-1)
        day = env[index]
        day[0]['bat'] = np.random.randint(0, 3) * 5.0 # randomly chosen initial battery level
        # play an episode and observe each state, i.e. hour
        for i in (0, len(day) - 2):
            state_t = day[i]
            # apply e-greedy policy to avoid local minima
            # and balance between exploration and exploitation
            if random.random() <= epsilon:
                action = np.random.randint(0, 5)
            else:
                inp = np.array(list(state_t.values()))
                inp = np.expand_dims(inp, axis=0)
                q = model.predict(inp)
                action = np.argmax(q[0])
            # apply a chosen action and observe the next state
            reward, state_tp1 = act(state_t, action, day[i + 1])
            day[i+1] = state_tp1
            # add gained experience to replay memory
            exp_replay.remember([state_t, action, reward, state_tp1])
            # get a batch of experiences from replay memory and train the model
            inputs, targets = exp_replay.get_batch(model, num_inputs, batch_size=batch_size)
            batch_loss = model.train_on_batch(inputs, targets)
            train_loss += batch_loss
        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Reward {:.2f}".format(e, epochs, loss, reward))
        loss_hist.append(train_loss)
        # annealing epsilon
        epsilon *= .9
    return loss_hist

# apply trained model to test environment
def test(model, test_env):
    total_reward_untrained = 0.0
    total_reward_trained = 0.0
    last_bat_level = 0.0
    # calculate total reward without applying the model and where is no battery
    for i in range(0, len(test_env)):
        day = test_env[i]
        for j in range(0, len(day)):
            state = day[j]
            total_reward_untrained += state['price'] * (state['gen'] - state['reg'] - state['car'])

    # calculate total reward after applying the model and using battery
    for i in range(0, len(test_env)):
        day = test_env[i]
        day[0]['bat'] = last_bat_level
        for j in range(0, len(day) - 1):
            state_t = day[j]
            inp = np.array(list(state_t.values()))
            inp = np.expand_dims(inp, axis=0)
            q = model.predict(inp) 
            action = np.argmax(q[0])
            reward, state_tp1 = act(state_t, action, day[j + 1])
            day[j+1] = state_tp1
            total_reward_trained += reward
        last_bat_level = day[len(day) - 1]['bat']

    # compare both rewards
    saving_p = abs((total_reward_trained - total_reward_untrained) / total_reward_untrained) * 100
    print("Total electricity cost for season without HEMS: ${:.2f}".format(total_reward_untrained / 100))
    print("Total electricity cost for season with HEMS: ${:.2f}".format(total_reward_trained / 100))
    print("Total savings/gain percentage: {:.2f}%".format(saving_p))

# [day of week, time in hour, regular consumption, shiftable consumption,
# microgeneration, battery level, shift in hours, price]
num_inputs = 8
# [do nothing, discharge, recharge, shift, discharge + shift, recharge + shift]
num_actions = 6 
max_memory = 1000
hidden_size = 100
batch_size = 24
model = get_model(num_inputs, num_actions, hidden_size)
model.summary()
exp_replay = ExperienceReplay(max_memory=max_memory)
epoch = 5000

# train model on train environment which is household's historical data for May 1 - Oct 31 of 2015
train_env = init_env('../hems/data/preprocessed_dataset_Summer2015_3.csv')
hist = train(model, train_env, epoch, verbose=0)
print("Training finished")

# test model on test environment which is household's historical data for May 1 - Oct 31 of 2016
test_env = init_env('../hems/data/preprocessed_dataset_Summer2016_3.csv')
test(model, test_env)

# To make sure training loss decreases in iteration
plt.plot(hist)
plt.xlabel("Number of Epochs")
plt.ylabel("Training Loss")
plt.show()

            
        

    
