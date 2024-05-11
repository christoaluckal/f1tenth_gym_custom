from lunar_lander.lunar_lander import LunarLander
import numpy as np
import random

eval_env = LunarLander(continuous=True,enable_wind=False)
eval_batch = []
while len(eval_batch) < 1000:
    state = eval_env.reset()
    done = False
    while not done:
        action = eval_env.action_space.sample()
        next_state, reward, done, _, _ = eval_env.step(action)
        if random.uniform(0,1) < 0.3:
            # print(state,'\n')
            if type(state) != tuple:
                t = state.tolist()
                eval_batch.append(t) 
        state = next_state

for e in eval_batch:
    print(e)
eval_batch = np.array(eval_batch)
np.save('lunar_lander/eval_batch.npy',eval_batch)