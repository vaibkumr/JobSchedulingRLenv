# DQN observations:
- For 3,3,3 with 600 episodes and 200 long memory:
  In an experiment performed 500 times:
    - 62 times DQN was better than SJF
    - 134 times it was equally as good as SJF
    - 304 times it was worse
- 5,5,5 after 6k episodes and 200 long memory hardly ever performs better than SJF. It needs parameter tweaking and more training.
    - Maybe increase the learning ratE?
    - Definitely increase the memory from 200 to something like 2-3k

# DQN update
- DQN.ipynb is the main notebook
- Takes very long to train. Reminder: try using autojit
- model contains the trained Net
- eval_net.m and train_net.m are trained on 3,3,3 for 6k episodes (32 batch size, 200 memory only)
- eval_name and train_name are trained on 5,5,5 for 60k episodes (32 batch size, 200 memory only, increase it?)


# Job scheduler environment for RL
Simulate a job scheduling environment without preemption and two job instance types with variable amount of resource instances, job numbers and max job time.

HOWTO (test):
- Run environment.py
- Open state.png on browser
- keep pressing F5

The environment is shares almost the same structure as of the gym environments with minor changes.

- action space: If you set maximum time as 3 then action space is [1, 2, 3]. 1 means schedule job of time 1. In general max time = mt would yield action_space = range(1, mt+1)
- reward:
    - For every action that yields a non terminal state the reward is 0.
    - For every action that is wrong (action = 1 when there is no job of time 1) a reward of -10 is given
    - For others, reward = sigma((Wj+Tj)/Tj) where Wj and Tj are waiting and burt time of job j respectively.


an example:
```python
  env = scheduler() #Make environment object
  n, mt, mr = 3, 3, 3
  #n is number of jobs
  #mt is max time
  #mr is max resource instances
  env.make(n, mt, mr) #make() creates the assignments and initializes a random start state
  env.render() # Render the environment to be visible
  some_random_action = env.sample() #sample() returns a sample action
  _, reward, _, _ = env.step(some_random_action)
  print(f"Reward is: {reward}")
  env.render()
```
