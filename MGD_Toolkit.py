import sys 

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Config as CFG
import MGD_Merged as MGD
class Dot:
    x = 0.0
    y= 0.0
    def __new__(cls, *args, **kwargs):
        return super(Dot, cls).__new__(cls)
    def __init__(self, x = None, y = None):
        if x is not None and y is not None:
            self.x = x
            self.y = y
        elif x is None:
            self.x = x
            self.y = 0
        elif y is None:
            self.x = 0
            self.y = y
        else:
            self.x = 0
            self.y = 0
    def __str__(self):
        return f'({self.x}, {self.y})'
    def __repr__(self):
        return f'({self.x}, {self.y})'
        

def ShapeParcer(dots:np.array):

    mask = np.zeros((CFG.Nx, CFG.Ny))
    for i in range(len(dots) - 1):
        dx, dy = CFG.dx, CFG.dy
        x1, y1 = dots[i].x * dx, dots[i].y * dy
        x2, y2 = dots[i + 1].x * dx, dots[i + 1].y * dy
        x = np.linspace(x1, x2, int(abs(x2-x1)/dx))
        y = np.linspace(y1, y2, int(abs(y2-y1)/dy))
        mask[y.astype(int), x.astype(int)] = 1
    return mask
    
def ViewShape(Mask:np.array):
    plt.imshow(Mask)
    plt.show()
    
def main():
    DotsPair = []
    DotsPair.append(Dot(0.0, 0.0))
    DotsPair.append(Dot(0.1, 0.2))
    DotsPair.append(Dot(0.3, 0.4))
    Mask = ShapeParcer(DotsPair)
    print(*Mask)
    
if __name__ == '__main__':
    main()