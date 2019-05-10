from pipeline import pipeline
import matplotlib.pyplot as plt
import numpy as np


def vcurve_int(dir):
    seeds = [2301, 3990, 8490, 8408, 9084, 2736, 111, 9483, 345, 1220]

    x = []
    y = []
    for i in range(10):
        print('iteration number:',i)
        seed = seeds[i]
        x1,y1 = pipeline(True,'recompute','none',seed,dir)
        x.append(x1)
        y.append(y1)

    plt.scatter(x,y)
    x = np.linspace(0, 1, 100)
    plt.plot(x,x)
    plt.xlabel('Training Accuracy')
    plt.ylabel('Test Set Accuracy')
    plt.title('Integration Validation Plot')
    plt.show()


# vcurve_int("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data")