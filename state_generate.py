import numpy as np
import pandas as pd
from tabulate import tabulate


def zeros(rows, columns):
    return pd.DataFrame(np.zeros((rows, columns)))

def fix(a, b, r):
    """if both are zero, make any one non zero"""
    if not a and not b:
        if np.random.randint(1, 2):
            a = np.random.randint(1, r)
        else:
            b = np.random.randint(1, r)
        return a, b
    return a, b

class Jobs():
    """
    Generates the jobs.
    """
    def __init__(self, n=3, max_time=3, max_resource=3):
        """
        Generates and prints the starting states
        Number of resources are 2
        @param n is number of jobs
        @param max_time is maximum time
        @param max_resource is maximum resources
        """

        self.max_time = max_time
        self.n = n
        self.max_resource = max_resource
        self.jlist1 = []
        self.jlist2 = []
        self.backlog = dict(
                            zip([t for t in range(1, self.max_time+1)],
                            [0 for t in range(1,self.max_time+1)])
                        )

    def generate_state(self):
        jtime = np.random.randint(1, self.max_time+1, self.n)
        uniques = pd.DataFrame(jtime)[0].value_counts()
        resource1_req = np.random.randint(0, self.max_resource+1, self.n)
        resource2_req = np.random.randint(0, self.max_resource+1, self.n)
        index = 0
        for time in range(1, self.max_time+1):
            if time in uniques.index:
                self.backlog[time] += uniques[time] - 1
                uniques[time] = 0
                r1, r2 = resource1_req[index], resource2_req[index]
                r1, r2 = fix(r1, r2, self.max_resource+1)
                m1, m2 = self.generate_job_matrix(time, r1, r2)
                self.jlist1.append(m1)
                self.jlist2.append(m2)
                index += 1
            else:
                m1, m2 = self.empty_generate_job_matrix(time)
                self.jlist1.append(m1)
                self.jlist2.append(m2)
        self.rMatrix1, self.rMatrix2 = self.generate_resource_matrix()


    def empty_generate_job_matrix(self, time):
        matrix1 = zeros(self.n * self.max_time, self.max_resource)
        matrix2 = zeros(self.n * self.max_time, self.max_resource)
        return matrix1, matrix2

    def generate_resource_matrix(self):
        rows = self.n * self.max_time
        columns = self.max_resource
        rMatrix1 = zeros(rows, columns)
        rMatrix2 = zeros(rows, columns)
        return rMatrix1, rMatrix2

    def generate_job_matrix(self, time, r1, r2):
        rows = self.n * self.max_time
        columns = self.max_resource
        matrix1 = zeros(rows, columns)
        matrix1.iloc[0:time, 0:r1] = time
        matrix2 = zeros(rows, columns)
        matrix2.iloc[0:time, 0:r2] = time
        return matrix1, matrix2

    def isempty_job(self, m1, m2):
        return ((not m1.any().any()) and (not m2.any().any()))

    def getState(self):
        self.generate_state()
        keys = ['resource1', 'resource2']
        for i in range(1, self.max_time+1):
            keys.append(f'job{i}')
        keys.append('backlog')
        values = [self.rMatrix1, self.rMatrix2]
        for m1, m2 in zip(self.jlist1, self.jlist2):
            subKeys = ['r1', 'r2']
            subValues = [m1, m2]
            values.append(dict(zip(subKeys, subValues)))
        values.append(self.backlog)
        return dict(zip(keys, values))

    def print(self):
        i = 1
        empty_counter = 0
        for m1, m2 in zip(self.jlist1, self.jlist2):
            m1.index += 1
            m2.index += 1
            empty = ""
            if self.isempty_job(m1, m2):
                empty_counter += 1
                empty = "[EMPTY]"
            print(f"Job {i} {empty}:")
            print(tabulate(m1, headers='keys', tablefmt='psql'),'\n')
            print(tabulate(m2, headers='keys', tablefmt='psql'))
            i += 1
        print("RESOURCE MATRIX: ")
        print(tabulate(self.rMatrix1, headers='keys', tablefmt='psql'))
        print(tabulate(self.rMatrix2, headers='keys', tablefmt='psql'))
        print("\n\nJobBacklog")
        print(tabulate(
                        [list(self.backlog.values())],
                        headers=list(self.backlog.keys()),
                        tablefmt='psql')
                        )
        print(f"# EMPTY JOBS: {empty_counter} \n# JOBS IN BACKLOG: {sum(list(self.backlog.values()))}")
