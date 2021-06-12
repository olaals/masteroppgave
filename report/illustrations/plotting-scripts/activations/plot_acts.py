import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-3,3,1000)
print(x)


relu = lambda x: max(0.0,x)


y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = relu(x[i])

sig = 1/(1 + np.exp(-x))


fig,ax = plt.subplots(1,2)

ax[0].plot(x,y)
ax[1].plot(x,sig)
plt.grid('on')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sigmoid = np.vectorize(sigmoid) #vectorize function
values=np.linspace(-10, 10) #generate values between -10 and 10
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(values, sigmoid(values), color=(1,0.25,0))
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = max(0.0, x[i])
    return y

values=np.linspace(-10, 10, 1000) #generate values between -10 and 10
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(values, relu(values), color=(1,0.25,0))
plt.show()
