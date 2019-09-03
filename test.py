import random
from env import Env


# target:取值有:'state-section'代表用断面；
# 'state-voltage'代表潮流状态调整的电压
# 'convergence'代表潮流收敛

# rand:代表环境reset()的时候是否需要随机reset，默认是不随机，不随机的话相当于只对一个样例在进行训练

# train:代表是否使用训练集
env = Env(
	target='state-section',
	rand=False,
	train=True
)

# 获得动作空间
n_actions = env.action_space
steps_done = 0
for epoch in range(100):
    # reset
    env.reset()
    # 获取当前的state状态
    state = env.get_state()
    cnt = 0
    while cnt < 5:
        cnt += 1
        action = random.randrange(n_actions)
        # 传入一个action,得到执行这个动作之后的state, reward, done
        next_state, reward, done = env.step(action)
        if done:
            break
