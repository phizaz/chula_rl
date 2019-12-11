# Online arena 

## Running the server

```
python arena_server.py
```


## Client-side 

See `test_onlineenv.ipnb` for example. 

```
from chula_rl.online.client import OnlineEnv

# to check occupied room see http://203.150.243.248:5000

room_id = 3 # any available room
env = OnlineEnv('http://203.150.243.248:5000', room_id)

s = env.reset()

actions = env.legal_moves()

ss, r, done, _ = env.step(actions[0])
```