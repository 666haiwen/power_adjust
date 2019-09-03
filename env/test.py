import random
from env import Env


env = Env(rand=True, target='convergence')
n_actions = env.action_space
steps_done = 0
for epoch in range(100):
    env.reset()
    state = env.get_state()
    cnt = 0
    while cnt < 5:
        cnt += 1
        action = random.randrange(n_actions)
        next_state, reward, done = env.step(action)
        if done:
            break
