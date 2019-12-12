# Online arena 



## Client-side 

See `test_onlineenv.ipnb` for example. 


### Play a single game

```
from chula_rl.online.client import OnlineEnv

# to check occupied room see http://203.150.243.248:5000

room_id = 3 # any available room
env = OnlineEnv('http://203.150.243.248:5000', room_id, play_match=False, superpower=False) # set the superpower

s = env.reset()

actions = env.legal_moves()

ss, r, done, _ = env.step(actions[0])
```

### Play a match game

Of 2 normal games, and 3 superpower games. 

Each player will take turn to begin. 

Only need to set `play_match=True` for the first player, the second player flags are ignored. 

```
from chula_rl.online.client import OnlineEnv

# to check occupied room see http://203.150.243.248:5000

room_id = 3 # any available room
env = OnlineEnv('http://203.150.243.248:5000', room_id, play_match=True) # need only for the first player

env.reset()
while True:
    actions = env.legal_moves()
    a = random.choice(actions)
    s, r, done, info = env.step(a)
    if done:
        env.reset()
```

## Running the server

You don't need to do this unless you want to. We have a running server at http://203.150.243.248:5000

```
python arena_server.py
```