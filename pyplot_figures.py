
import numpy as np
import matplotlib.pyplot as plt
from orb_calculator import E

def plot_rotation(el, angle, label, title=None, peak_idx=None):
    if not plt.isinteractive():
        plt.ion()
        plt.show()

    plt.ylabel('Energy (kcal/mol)')
    plt.xlabel('Angle (deg)')
    if title:
        plt.title(title)
    
    x = np.append(angle, 360) 
    y = (np.append(el,el[0])-el[0])*E
    plt.plot(x, y, label=label)
    plt.xlim(0,360)
    plt.ylim(0,20)

    if peak_idx is not None:
        plt.plot(x[peak_idx], y[peak_idx], 'x', color='black')

    plt.legend()
    plt.draw()
    plt.pause(0.001)
    

