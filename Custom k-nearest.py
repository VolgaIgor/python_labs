import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.neighbors
import random

class custom_knearest:
    def __init__(self, num_neighbors):
        self.n_neighbors = num_neighbors
    
    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x_new):
        if type(x_new) == int:
            x_new = np.array([[x_new]])
        elif type(x_new) == list:
            x_new = np.array(x_new)

        return_array = np.zeros((x_new.size, 1))
        for new_x_counter, new_x_value in enumerate(x_new):
            new_x_value = new_x_value[0]
            for counter, value in enumerate(self.x):
                if value[0] > new_x_value:
                    break
            l = []
            for i in range(self.n_neighbors):
                right = counter + i
                left = counter - i - 1
                if right < self.x.size:
                    l.append([abs(new_x_value - self.x[right][0]), right])
                if left >= 0:
                    l.append([abs(new_x_value - self.x[left][0]), left])
            l.sort()
            sum = 0
            for i in range(self.n_neighbors):
                sum += self.y[ l[i][1] ][0]
            return_array[new_x_counter][0] = sum / self.n_neighbors
        return return_array

def genRndPoints( min, max, count ):
    points = []
    for i in range(count):
        points.append( [
            random.randint(min, max),
            random.randint(min, max)
        ] )
    return np.array(points)

points = [[ 9054.914, 6.0],
       [ 9437.372, 5.6],
       [12239.894, 4.9],
       [12495.334, 5.8],
       [15991.736, 6.1],
       [17288.083, 5.6],
       [18064.288, 4.8],
       [19121.592, 5.1],
       [20732.482, 5.7],
       [25864.721, 6.5],
       [27195.197, 5.8],
       [29866.581, 6.0],
       [32485.545, 5.9],
       [35343.336, 7.4],
       [37044.891, 7.3],
       [37675.006, 6.5],
       [40106.632, 6.9],
       [40996.511, 7.0],
       [41973.988, 7.4],
       [43331.961, 7.3],
       [43603.115, 7.3],
       [43724.031, 6.9],
       [43770.688, 6.8],
       [49866.266, 7.2],
       [50854.583, 7.5],
       [50961.865, 7.3],
       [51350.744, 7.0],
       [52114.165, 7.5],
       [55805.204, 7.2]]
pdPoints = pd.DataFrame( points, columns=['x', 'y'] )
x = np.c_[pdPoints["x"]]
y = np.c_[pdPoints["y"]]

pdPoints.plot(kind='scatter', x="x", y='y')

model1 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model1.fit(x, y)

model2 = custom_knearest(3)
model2.fit(x, y)

X_test = np.arange(x[0][0], x[-1][0], 0.1)
X_test = X_test.reshape((X_test.shape[0], 1))

plt.plot(X_test, model1.predict(X_test), color="red", linewidth=4)
plt.plot(X_test, model2.predict(X_test), color="blue", linewidth=2)
plt.show()
