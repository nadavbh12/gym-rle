import gym
import gym_rle

env = gym.make('ClassicKong-v0')
env.reset()
env.render()

while True:
    action = 0
    env.step(action)
    env.render()
