import numpy as np
import pandas as pd
from tabulate import tabulate
from state_generate import Jobs
from render_image import Render
import copy
import gym


def cut_matrix(matrix, mr, mt):
    for n in range(mt):
        if matrix[n : n+mr].all() == 0:
            time =  n
    return matrix[:n * mr]

def zero_df(rows, columns):
    """ returns dataframe filled with zeros and size rows x columns """
    df = pd.DataFrame(np.zeros((rows, columns)))
    return df

def get_empty_job(rows, columns):
    """ generates an empty job of size rows x columns """
    matrix = zero_df(rows, columns)
    return matrix, matrix

def isempty_job(m1, m2):
    """ checks if the job having m1, m2 requirement matrix is empty """
    return ((not m1.any().any()) and (not m2.any().any()))

def get_index(s):
    """ returns the index of first zero in the pd.Series """
    return len(s) - len(s[s==0])

def get_d(df1, df2):
    """returns the index of first zero in a row """
    s1, s2 = df1.iloc[0], df2.iloc[0]
    return ((len(s1)-len(s1[s1==0])), (len(s2)-len(s2[s2==0])))

def make_job(time, rows, columns):
    """create a new job of time: time and rows x columns representation"""
    r1, r2  = np.random.randint(0, columns, 2)
    if r1==0 and r2==0: #Can't make both zero
        if np.random.randint(0, 2):
            r1 = 1
        else:
            r2 = 1
    matrix1 = zero_df(rows, columns)
    matrix1.iloc[0:time, 0:r1] = time
    matrix2 = zero_df(rows, columns)
    matrix2.iloc[0:time, 0:r2] = time
    return matrix1, matrix2

class Scheduler(gym.Env):
    def __init__(self, n, mt, mr):
        """
        Simulate a job scheduling environment without preemption and two job
        instance types with variable amount of resource instances, job numbers
        and max job time.
        @param n number of jobs
        @param mt max burst time
        @param mr max resource instance requirement
        There are two resources
        """
        self.state = 'INVALID' #Call make() to make an environment
        self.action_sequence = [] #No actions taken yet
        self.invalid_action_reward = -100
        self.valid_action_reward = 0 #Try changing this
        self.n = n
        self.mt = mt
        self.mr = mr
        self.action_space = [i for i in range(1, mt+1)]
        self.action_space_n = len(self.action_space)
        self.observation_space = self.get_observation_space()
        self.symbol = [str(i) for i in range(1, mt+1)]
        self.rows = n * mt
        self.columns = mr
        self.reset()

    def sample(self):
        """return a sample action"""
        return np.random.choice(self.action_space)

    def reset(self):
        """reset the environment"""
        self.done = False
        self.cr = 0
        self.state = self.generate_start_state()
        self.reward = 0
        return self.collect_matrices()

    def setPosition(self, state):
        """sets current state to some saved state"""
        self.state = state
        self.reward = 0

    def generate_start_state(self):
        """generates a new start state"""
        jObject = Jobs(self.n, self.mt, self.mr)
        self.state = jObject.getState()
        self.BACKLOG = self.state['backlog'].copy()
        return(self.state)

    def load_from_backlog(self, time):
        """load a job from the backlog"""
        if not self.state['backlog'][time]:
            return False #No job in backlog
        self.state['backlog'][time] -= 1
        instance = self.BACKLOG[time] - self.state['backlog'][time]
        symbol = f"{time}.{instance}"
        self.symbol[time-1] = symbol
        m1, m2 = make_job(time, self.rows, self.columns)
        self.state[f'job{time}'] = dict(zip(['r1', 'r2'], [m1, m2]))
        # print("SYMBOLS: ", symbol,'\n',self.BACKLOG,'\n', self.state['backlog'])
        return True

    def sched_job(self, time):
        """schedule the job at time: @param time"""
        assert self.state != "INVALID", "Invalid state, reset the environment"
        m1, m2 = self.state[f'job{time}']['r1'], self.state[f'job{time}']['r2']
        if isempty_job(m1, m2):
            self.state = 'INVALID'
            self.done = True
            return self.invalid_action_reward
        else:
            self.state[f'job{time}']['r1'], self.state[f'job{time}']['r2'] = \
                                    get_empty_job(self.rows, self.columns)
            self.state['resource1'], self.state['resource2'] = \
                                    self.allocate_job(time, m1, m2)
            reward = self.get_reward()
            self.load_from_backlog(time)
            return reward

    def allocate_job(self, time, m1, m2):
        """allocate the job at time: time"""
        r1, r2 = self.state['resource1'], self.state['resource2']
        d1, d2 = get_d(m1, m2)
        r1, r2 = self.fill_matrix(r1, r2, time, d1, d2)
        return r1, r2

    def fill_matrix(self, r1, r2, time, d1, d2):
        """
        fills the resource matrices r1 and r2 with the job having time: time
        and resource requirement d1 and d2
        """
        symbol = self.symbol[time-1]
        l1 = [] #l1 is the list containing the number of empty boxes in each row
        l2 = [] #l2 is the list containing the number of empty boxes in each row
        update_cr = False
        for row in r1.itertuples(index=False):
            l1 += [list(row).count(0)]
        for row in r2.itertuples(index=False):
            l2 += [list(row).count(0)]
        row = self.cr
        if l1[row] < d1 or l2[row] < d2:
            update_cr = True
        while(row<self.rows and time>0):
            if l1[row] >= d1 and l2[row] >= d2:
                if update_cr:
                    update_cr = False
                    self.cr = row
                zeroIndex1 = get_index(r1.iloc[row])
                zeroIndex2 = get_index(r2.iloc[row])
                if d1:
                    r1.iloc[row, zeroIndex1:zeroIndex1+d1] = symbol
                if d2:
                    r2.iloc[row, zeroIndex2:zeroIndex2+d2] = symbol
                time -= 1
            row += 1
        return r1, r2

    def sjf(self, save_state = True):

        if save_state:
            saved_state = copy.deepcopy(self.state)
        done = False
        reward = 0
        while not done:
            for time in range(1, self.mt + 1):
                m1, m2 = self.state[f'job{time}']['r1'], self.state[f'job{time}']['r2']
                if isempty_job(m1, m2):
                    continue
                s, r, done, info = self.step(time)
                reward += r
        if save_state:
            self.state = copy.deepcopy(saved_state)
            self.done = False
        return reward


    def all_done(self):
        """"returns true if all jobs have been scheduled"""
        state = self.state
        for time in range(1, self.mt+1):
            key = f"job{time}"
            # print("PRINTING STATE\n\n")
            # print(state)
            m1, m2 = state[key]['r1'], state[key]['r2']
            if not isempty_job(m1, m2):
                return False
        if not all(v == 0 for v in state['backlog'].values()):
            return False
        self.done = True
        return True

    def isrunning(self):
        """returns true if final state not reached"""
        if self.done:
            return False
        return True

    def fetch_reward(self, m):
        """calculate reward helper function"""
        l = [] #Contains symbols of all the jobs for which calculations are done
        rewards = []
        for row in range(0, self.rows):
            waiting_time = row
            for jname in m.iloc[row].unique():
                #example jname are 1.11, 10.11, 2, 1, 1.1
                if jname not in l and int(str(jname).split('.')[0]):
                    l.append(jname)
                    burst_time = int(str(jname)[0])
                    rewards.append((waiting_time + burst_time) / burst_time)
        return sum(rewards)

    def calculate_reward(self):
        """calculate reward helper function"""
        if type(self.state) is str:
            return -1
        m1 = self.state['resource1']
        m2 = self.state['resource2']
        r1, r2 = self.fetch_reward(m1), self.fetch_reward(m2)
        return max(r1, r2)

    def get_reward(self):
        """Get reward from state"""
        state = self.state
        if state == "INVALID":
            return self.invalid_action_reward
        if self.all_done():
            return -self.calculate_reward() #Negative of waited burst time
        else:
            return self.valid_action_reward

    def step(self, action):
        """takes the "action" step in the environment
        @param action is the action to be taken example 1 means schedule job
        with time 1 units
        returns :
        @environment: state
        @reward: reward for the state, action pair
        @self.done: true if end of episode
        @infor: sequence of actions taken till now
        """
        assert action in self.action_space, "Invalid action"
        self.action_sequence.append(action)
        reward = self.sched_job(action)
        environment = self.state
        info = self.action_sequence
        flat_state = self.collect_matrices()
        return flat_state, reward, self.done, info

    def render(self, terminal=True):
        """
        @terminal if True then prints the state on terminal else,
        displays the current state as colored image file 'state.png'
        """
        if terminal:
            self.render_terminal()
        elif self.state == "INVALID":
            print("invalid action")
            print("DONE")
            Render(self.state, self.n).render_invalid()
            return
        else:
            Render(self.state, self.n).render()

    def get_observation_space(self):
        some_state = self.generate_start_state()
        matrices = np.array([])
        for key, value in some_state.items():
            if key in ['resource1', 'resource2']:
                # matrices += np.array(value).reshape(-1)
                matrices = np.append(matrices, np.array(value).reshape(-1))
            if key[0:-1] == 'job':
                m1 = cut_matrix(np.array(value['r1']).reshape(-1), self.mr, self.mt)
                m2 = cut_matrix(np.array(value['r2']).reshape(-1), self.mr, self.mt)
                # matrices = np.append(matrices, np.array(value['r1']).reshape(-1))
                # matrices = np.append(matrices, np.array(value['r2']).reshape(-1))
                matrices = np.append(matrices, m1)
                matrices = np.append(matrices, m2)
            if key == 'backlog':
                matrices = np.append(matrices, np.array([list(value.values())]))
                # matrices += np.array([list(value.values())])
        return matrices.size

    def collect_matrices(self):
        """collect the matrices and send them to flaten"""
        matrices = np.array([])
        if self.state == "INVALID":
            return -1 * np.ones(self.observation_space)
        for key, value in self.state.items():
            if key in ['resource1', 'resource2']:
                # matrices += np.array(value).reshape(-1)
                matrices = np.append(matrices, np.array(value).reshape(-1))
            if key[0:-1] == 'job':
                m1 = cut_matrix(np.array(value['r1']).reshape(-1), self.mr, self.mt)
                m2 = cut_matrix(np.array(value['r2']).reshape(-1), self.mr, self.mt)
                # matrices = np.append(matrices, np.array(value['r1']).reshape(-1))
                # matrices = np.append(matrices, np.array(value['r2']).reshape(-1))
                matrices = np.append(matrices, m1)
                matrices = np.append(matrices, m2)
            if key == 'backlog':
                matrices = np.append(matrices, np.array([list(value.values())]))
                # matrices += np.array([list(value.values())])
        self.observation_space = matrices.size
        return matrices.astype(float)

    def render_terminal(self):
        """Displays the current state in ASCII on terminal"""
        if self.state == "INVALID":
            print("invalid action")
            print("DONE")
            return
        for key, value in self.state.items():
            print(key)
            if key in ['resource1', 'resource2']:
                print(tabulate(value, headers='keys', tablefmt='psql'))
            if key[0:-1] == 'job':
                print(tabulate(value['r1'], headers='keys', tablefmt='psql'))
                print(tabulate(value['r2'], headers='keys', tablefmt='psql'))
            if key == 'backlog':
                print(tabulate(
                                [list(value.values())],
                                headers=list(value.keys()),
                                tablefmt='psql')
                                )
        if self.done:
            print("DONE")
        else:
            print("NOT DONE")
