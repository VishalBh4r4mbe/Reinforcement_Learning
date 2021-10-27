import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

class Agent(object):
	def __init__(self,env,gamma=0.98,alpha=0.1):
		self.env=env
		self.state_size=env.observation_space.n							#Number of possible states/configurations 
		self.action_size=env.action_space.n								#Number of possible actions to take 
		self.gamma=gamma												#Discount factor to take future rewards into acount
		self.alpha=alpha												#Learning rate, to update the Q-table
		self.n_hl1=32													#Number of units in the first hidden layer
		self.n_hl2=32													#Number of units in the seconf hidden layer
		self.action_space=np.arange(self.action_size)					#Defining the array of the possible actions the agent can take
		self.network=self.build_model()									#Initializing the Q-Table
		self.reward_history=[]											#To maintain track of average reward per episode while training
		self.episode_lengths=[]											#To keep track of the length od episodes

	def build_model(self):

		inputs=Input(shape=[self.state_size,])
		X=Dense(self.n_hl1)(inputs)
		X=Dense(self.n_hl2)(X)
		outputs=Dense(self.action_size)(X)
		model=Model(inputs=inputs,outputs=outputs)
		model.compile(optimizer=Adam(learning_rate=self.alpha),loss="categorical_crossentropy")
		return model

	def epsilon_greedy_action(self,state,epsilon=0.2):

		state=np.eye(self.state_size)[state]
		qvalues=self.network.predict(state.reshape([1,self.state_size]))
		A=np.zeros((qvalues.shape[1]))+epsilon/self.action_size
		greedy_action=np.argmax(qvalues[0])
		A[greedy_action]+=1-epsilon
		action=np.random.choice(self.action_space,p=A)
		return action

	def target_qvalues(self,qvalues,action,next_state,reward):

		next_state=np.eye(self.state_size)[next_state]
		q_nextstate=self.network.predict(next_state.reshape([1,self.state_size]))
		max_q=np.argmax(q_nextstate[0])
		target_qvalues=qvalues
		target_qvalues[action]=reward+self.gamma*q_nextstate[0,max_q]
		return target_qvalues

	def update_network(self,state,action,reward,next_state):

		state=np.eye(self.state_size)[state]
		qvalues=self.network.predict(state.reshape([1,self.state_size]))
		target_qvalues=self.target_qvalues(qvalues[0],action,next_state,reward).reshape([1,self.action_size])
		state=state.reshape([1,self.state_size])
		self.network.fit(state,target_qvalues,epochs=1,verbose=0)

	#Training the agent through a series of episodes
	def train(self,num_episodes=10000):

		#Iterating over a number of episodes
		for i in tqdm(range(num_episodes)):
			reward_buffer=0
			j=0
			state_now=self.env.reset()
			while True:
				action=self.epsilon_greedy_action(state_now)
				state_next,reward,done,_=self.env.step(action)
				reward_buffer+=reward
				j+=1
				self.update_network(state_now,action,reward,state_next)
				if done==True:
					self.reward_history.append(reward_buffer/j)
					self.episode_lengths.append(j)
					if (i+1)%100==0:
						print("Average reward claimed by the agent in episode {} : {}".format(i+1,self.reward_history[-1]))
						print("Length of episode {} : {}".format(i+1,self.episode_lengths[-1]))
					break
				else:
					state_now=state_next

env=gym.make("FrozenLake-v0")

agent=Agent(env=env)
agent.train()

reward_history=agent.reward_history

sfig, axs = plt.subplots(1,2)
axs[0].plot(agent.reward_history)
axs[0].set_title('Average Reward per Episode')
axs[1].plot(agent.episode_lengths, 'tab:orange')
axs[1].set_title('Episode_Length')

plt.show()

print('Average Reward throughout all episodes={}'.format(sum(reward_history)/len(reward_history)))