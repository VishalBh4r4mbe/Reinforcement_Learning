import gym
from gym import  wrappers
import numpy as np

env  = gym.make('FrozenLake-v0')
Q = np.random.rand(env.observation_space.n, env.action_space.n)
countSA = np.zeros([env.observation_space.n, env.action_space.n])

episodes = 1000000
epsilon = 0.5
rewardList = []
for i in range(episodes):
    state = env.reset()
    done = False
    resultList = []
    epReward = 0
    if(i%100 == 0 and i!=0):
        epsilon = epsilon*0.9995
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        newState, reward, done, info = env.step(action)
        # if(done==True and newState!=15):
        #     reward = -0.1

        resultList.append((state,action))
        state = newState   
        epReward+=reward
    rewardList.append(epReward)
    for state, action in resultList:
        countSA[state, action] += 1
        Q[state, action] = Q[state, action] + (1.0/countSA[state, action])*(reward - Q[state, action])
    if i % 100 == 0:
        print("Success Rate = ",np.mean(rewardList)," Epsilon = ",epsilon)
print(np.argmax(Q, axis=1))
print("Average Reward: {}".format(np.mean(rewardList)))
print("Std Reward: {}".format(np.std(rewardList)))
env.close()
