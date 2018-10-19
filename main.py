
import tensorflow as tf
import numpy as np
import time
from replay_buffer import ReplayBuffer
from q_network import Network
import gym

DEVICE = '/gpu:0'

# Base learning rate 
LEARNING_RATE = 1e-4
RANDOM_SEED = 1234

def to_one_hot(state,state_dim):
        one_hot = np.zeros(state_dim)
        one_hot[state] = 1.
        return one_hot

def trainer(epochs=1000, MINIBATCH_SIZE=32, GAMMA = 0.99,save=1, save_image=1, epsilon=1.0, min_epsilon=0.05, BUFFER_SIZE=15000, train_indicator=True, render = True):
    with tf.Session() as sess:

        # configuring the random processes
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
       
        # environment
        
        env = gym.make('CartPole-v1') 
        print('action ', env.action_space)
        print('obs ', env.observation_space)
        observation_space = 4
        action_space = 2
        '''
        env = gym.make('FrozenLake8x8-v0') 
        print('action ', env.action_space)
        print('obs ', env.observation_space)
        observation_space = 64
        action_space = 4
        '''
        # agent
        agent = Network(sess,observation_space, action_space,LEARNING_RATE,DEVICE,layer_norm=False)
        
        # TENSORFLOW init seession
        sess.run(tf.global_variables_initializer())
               
        # Initialize target network weights
        agent.update_target_network()
        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        replay_buffer.load()
        print('buffer size is now',replay_buffer.count)
        # this is for loading the net  
        loss = 0.
        if save:
            try:
                agent.recover()
                print('********************************')
                print('models restored succesfully')
                print('********************************')
            except: 
                print('********************************')
                print('Failed to restore models')
                print('********************************')
        
        for i in range(epochs):

            if (i%500 == 0) and (i != 0): 
                print('*************************')
                print('now we save the model')
                agent.save()
                #replay_buffer.save()
                print('model saved succesfuly')
                print('*************************')
            
            if i%200 == 0: 
                 agent.update_target_network()
                 print('update_target_network')
            
            state = env.reset()
            # state = to_one_hot(state, observation_space)
            # print('state', state)
            q0 = np.zeros(action_space)
            ep_reward = 0.
            done = False
            step = 0
            
            while not done:
                
                epsilon -= 0.00001
                epsilon = np.maximum(min_epsilon,epsilon)
                
                # Get action with e greedy
                
                if np.random.random_sample() < epsilon:
                    #Explore!
                    action = np.random.randint(0,action_space)
                else:
                    # Just stick to what you know bro
                    q0 = agent.predict(np.reshape(state,(1,observation_space)) ) 
                    action = np.argmax(q0)
                    # print(q0)

                next_state, reward, done, info = env.step(action)
                # next_state = to_one_hot(next_state, observation_space)

                # img = env.render()
                # I made a change to the reward
                reward = np.cos(2*next_state[3])
                               
                if train_indicator:
                    
                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:
                        # 4. sample random minibatch of transitions: 
                        s_batch, a_batch, r_batch, t_batch, s2_batch= replay_buffer.sample_batch(MINIBATCH_SIZE)
                        q_eval = agent.predict_target(np.reshape(s2_batch,(MINIBATCH_SIZE,observation_space)))
                        q_target = np.zeros(MINIBATCH_SIZE)
                        # q_target = q_eval.copy()
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                q_target[k] = r_batch[k]
                            else:
                                q_target[k] = r_batch[k] + GAMMA * np.max(q_eval[k])

                        #5.3 Train agent! 
                        loss, _ = agent.train(np.reshape(a_batch,(MINIBATCH_SIZE,1)),np.reshape(q_target,(MINIBATCH_SIZE, 1)), np.reshape(s_batch,(MINIBATCH_SIZE,observation_space)) )
                        # print('loss', loss)
                        #print('acciones', np.reshape(a_batch,(MINIBATCH_SIZE)))
                        #actiones, action_one_hot, out, target_q_t, q_acted_0, q_acted, delta, loss, _ = agent.train_v2(np.reshape(a_batch,(MINIBATCH_SIZE,1)),np.reshape(q_target,(MINIBATCH_SIZE, 1)), np.reshape(s_batch,(MINIBATCH_SIZE,observation_space)) )
                        #print('action',actiones, 'action one hot', action_one_hot, 'out', out,'q acted 0', q_acted_0,  'q acted', q_acted, 'target', target_q_t, 'loss',loss, 'delta', delta)
                # 3. Save in replay buffer:
                replay_buffer.add(state,action,reward,done,next_state) 
                
                # prepare for next state
                state = next_state
                ep_reward = ep_reward + reward
                step +=1
                
                               
            
            
            print('th',i+1,'Step', step,'Reward:', round(ep_reward,0),'epsilon', round(epsilon,3), 'loss', loss )
            # print('q0', q0)

        print('*************************')
        print('now we save the model')
        agent.save()
        #replay_buffer.save()
        print('model saved succesfuly')
        print('*************************')
        
        


if __name__ == '__main__':
    trainer(epochs=2000 ,save_image = False, epsilon= 1., train_indicator = True)
