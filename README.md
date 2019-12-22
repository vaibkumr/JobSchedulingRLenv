# JobScheduler
This is a job scheduling environment for training agents through reinforcement learning algorithms like DQN.
Simulate a job scheduling environment without preemption and two job instance types with variable amount of resource instances, job numbers and max job time. This environment is created by extending the [openAI gym](https://gym.openai.com/docs/) class for compatibility.

# How?
```python
from scheduler import Scheduler
env = Scheduler(n=3, mt=3, mr=3)
some_random_action = env.sample() #sample() returns a randomly sampled action
state, reward, done, observation = env.step(some_random_action)
print(f"Reward is: {reward}")
if done:
  print(f"Episode done, call reset().")
env.render()
```
- `n`: Number of jobs
- `mt`: Maximum burst time
- `mr`: Maximum resource instance requirement (Note: There are only 2 types of instances. The requirement of each instance can be variable and controlled by `mr`)

# Action space and reward function
- **Action space**: If you set maximum time as 3 (`mt=3`) then action space is `[1, 2, 3]`. 1 means schedule job of time 1. In general, `mt` would yield `action_space = range(1, mt+1)`
- **Reward**:
    - For every action that yields a non terminal state the reward is 0.
    - For every action that is wrong (e.g. action = 1 when there is no job of time 1 left) a heavily negative reward of -100 is given (Try changing these in the code as per your needs)
    - For others, `reward = sigma((Wj+Tj)/Tj)` where `Wj` and `Tj` are waiting and burst time of job j respectively.

# State representation
A state looks like this:
![](https://i.imgur.com/dJwtVNF.png)

(This image is created using `env.render(False)`. However, if you use large values of `n(>4)` and `mr(>5)` use `env.render(True)` to render the state on terminal and not as images using matplotlib)

Here's what this image represents:
- The six group of matrices on top are resource type 1.
- The six group of matrices at the bottom are resource type 2 (there can only be 2 resource types for now).
- `n^{th}` matrix in any of these groups represents the job with burst time `n` (start counting from 1 not 0).
- The matrix on the middle right is the backlog. `n` darkened elements at the `k^{th}` row means `n` jobs with burst time of `k` are waiting.
- In this particular example, there are 4 jobs: Green, Blue, yellow and a fourth in the waiting queue with burst time of 2, 3, 4 and 2 respectively.

Internally, all of this is represented as matrices (internal environment state is this matrix in the row-wise order). This is a continuous state environment and you need neural networks to deal with these (most likely).
