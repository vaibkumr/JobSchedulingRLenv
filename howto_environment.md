# This all needs to be there in the environment class:
Functions and variables:
1. __init__(self, n=3, mt=5, mr=3)
2. setStateMatrix
3. setRewardMatrix
4. setPosition(self, state): Sets the agent in a prticular state
5. render
6. reset: Reset environment and set into a randomly chosen starting state
7. step(self, action): it moves a step action. Returns:
    -> observation (object): observations about environment like velocity etc
    -> reward (float)
    -> done (boolean)
    -> infor (dict) diagnostic information example raw probability etc
8. var env.action_space
9. var env.observation_space    

from environment import scheduler
a = scheduler()
a.make(5,5,4)
a.render()
