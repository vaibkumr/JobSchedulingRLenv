import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure

DPI = 30
FIGSIZE = (8, 10)
FILENAME = 'state.png'
CMAP = {
        "0" : "w",
        "0.0" : "w",
        '1' : "#FF0000",
        '2' : "#00FF00",
        '3' : "#0000FF",
        '4' : "#FFFF00",
        '5' : "#FF00FF",
        '1.0' : "#FF0000",
        '2.0' : "#00FF00",
        '3.0' : "#0000FF",
        '4.0' : "#FFFF00",
        '5.0' : "#FF00FF",
        '1.1' : "#00FFFF",
        '1.2' : "#008080",
        '1.3' : "#008000",
        '1.4' : "#800000",
        '1.5' : "#2378CE",
        '2.1' : "#4FAD99",
        '2.2' : "#452423",
        '2.3' : "#5ED7A1",
        '2.4' : "#800000",
        '2.5' : "#2378CE",
        '3.1' : "#c0fa36",
        '3.2' : "#f1c2a5",
        '3.3' : "#6cc988",
        '3.4' : "#800000",
        '3.5' : "#2378CE",
        '4.1' : "#a14e47",
        '4.2' : "#7b56ef",
        '4.3' : "#0715eb",
        '4.4' : "#800000",
        '4.5' : "#2378CE",
        '5.1' : "#7bd4c3",
        '5.2' : "#980fb6",
        '5.3' : "#7c2f87",
        '5.4' : "#800000",
        '5.5' : "#2378CE",
}

def get_colors(matrices):
    global CMAP
    cs = []
    for m in matrices:
        cs.append(np.array(m.applymap(lambda x: CMAP[str(x)])))
    return cs


class Render():
    def __init__(self, state, n):
        self.n = n #number of jobs
        self.state = state

    def render_invalid(self):
        figure(num=None, figsize=FIGSIZE, dpi=DPI)
        plt.savefig(FILENAME, bbox_inches='tight')
        plt.close()

    def render(self):
        state = self.state
        r1, r2 = state['resource1'], self.state['resource2']
        rows = 2
        columns = len(state.keys())
        plot_matrix = (rows, columns)
        i = 2
        figure(num=None, figsize=FIGSIZE, dpi=DPI)
        for key in state.keys():
            if key == 'resource1':
                colors = get_colors([state[key]])[0]
                self.plot([2, columns, 1], [colors])
            elif key == 'resource2':
                # location = f"2{columns}{columns+1}"
                colors = get_colors([state[key]])[0]
                self.plot([2, columns, columns+1], [colors])
            elif key == 'backlog':
                self.plot_backlog(state[key], [2, columns, columns])
            else:
                m1, m2 = state[key]['r1'], state[key]['r2']
                # location1, location2 = f"2{columns}{i}", f"2{columns}{i+columns}"
                colors1, colors2 = get_colors([m1, m2])
                self.plot([[2, columns, i], [2, columns, columns+i]], [colors1, colors2])
                i += 1
        # plt.show()
        plt.savefig(FILENAME, bbox_inches='tight')
        plt.close()

    def plot(self, locations, colors):
        if type(locations[0]) is not list:
            ax = plt.subplot(locations[0], locations[1], \
                            locations[2], frameon=False)
            ax.axis('off')
            ax.table(cellColours=colors[0], loc='center')
        else:
            for l, c in zip(locations, colors):
                ax = plt.subplot(l[0], l[1], l[2], frameon=False)
                ax.axis('off')
                ax.table(cellColours=c, loc='center')

    def plot_backlog(self, backlog, locations):
        rows = len(self.state.keys()) - 3
        columns = self.n - 1
        color = "#000000"
        print((rows, columns))
        colors = np.full((rows, columns), '#FFFFFF', dtype='U7')
        for key in backlog.keys() :
            print(backlog)
            colors[key-1][0:backlog[key]] = color
        ax = plt.subplot(locations[0], locations[1], \
                        locations[2], frameon=False)
        ax.axis('off')
        ax.table(cellColours=colors, loc='center')
