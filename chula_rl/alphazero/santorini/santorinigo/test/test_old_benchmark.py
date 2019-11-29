from chula_rl.alphazero.santorini.santorinigo.environment import Santorini
import random
import time

env = Santorini()
start_time = time.time()
state = env.reset()
for i_act in range(1000):
    actions = env.legal_moves()
    a = random.choice(actions)
    s, r, done, info = env.step(a)
    if done:
        env.reset()
end_time = time.time()
print((end_time - start_time) / 1000, 'seconds')