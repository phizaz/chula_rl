from chula_rl.alphazero.santorini.santorinigo.fast import Santorini
import random
import time

env = Santorini(superpower=False)
start_time = time.time()
state = env.reset()
n = 10000
for i_act in range(n):
    actions = env.legal_moves()
    # if len(actions) == 0:
    #     env.reset()
    #     continue
    #     print('no action')
    #     print('current:', env.current_player)
    #     break
    a = random.choice(actions)
    s, r, done, info = env.step(a)
    if done:
        env.reset()
end_time = time.time()
print((end_time - start_time) / n, 'seconds')