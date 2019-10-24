from chula_rl.policy.wrapper.for_each_data import ForEachData


class Policy:
    def optimize_step(self, data):
        print(data)


policy = Policy()
policy = ForEachData(policy)
policy.optimize_step({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
})