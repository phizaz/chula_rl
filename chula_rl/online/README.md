# Online arena 



## Client-side 

See `test_onlineenv.ipnb` for example. 

```
from chula_rl.online.client import OnlineEnv

# to check occupied room see http://203.150.243.248:5000

room_id = 3 # any available room
env = OnlineEnv('http://203.150.243.248:5000', room_id, superpower=False) # set the superpower

s = env.reset()

actions = env.legal_moves()

ss, r, done, _ = env.step(actions[0])
```

## Running the server

You don't need to do this unless you want to. We have a running server at http://203.150.243.248:5000

```
python arena_server.py
```