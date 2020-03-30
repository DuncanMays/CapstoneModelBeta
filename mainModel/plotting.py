import io, json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def serializeNumpyArray(array):
	memFile = io.BytesIO()
	np.save(memFile, array)
	memFile.seek(0)
	return json.dumps(memFile.read().decode('latin-1'))

def deserializeNumpyArray(string):
	memFile = io.BytesIO()
	memFile.write(json.loads(string).encode('latin-1'))
	memFile.seek(0)
	return np.load(memFile, allow_pickle=True)

f = open('modelResults', 'r')
dataStr = f.read()
f.close()

dataObj = json.loads(dataStr)

for i in dataObj:
	if isinstance(dataObj[i], str):
		dataObj[i] = deserializeNumpyArray(dataObj[i])

avgRewards = dataObj['avgRewards']
demand = dataObj['demand']
production = dataObj['production']
prodPrice = dataObj['prodPrice']
retailPrice = dataObj['retailPrice']
solarProduction = dataObj['solarProduction']
windProduction = dataObj['windProduction']
gasProd = dataObj['gasProd']
avgActions = dataObj['avgActions']

T = np.arange(0, 24, 2)

# dailyGasProd = []
# for i in range(int(len(gasProd)/len(T))):
# 	dailyGasProd.append(sum(gasProd[i*len(T): (i+1)*len(T)]))

# plt.plot(range(len(dailyGasProd)), dailyGasProd)
# plt.show()

fig, ax = plt.subplots()

breadth = 12

x = range(breadth)
prodLine, = ax.plot(x, production[0:breadth])
demLine, = ax.plot(x, demand[0:breadth])
gasLine, = ax.plot(x, gasProd[0:breadth])
actLine, = ax.plot(x, avgActions[0:breadth])
rewLine, = ax.plot(x, avgRewards[0:breadth])

def init():  # only required for blitting to give a clean slate.
	prodLine.set_ydata([np.nan] * len(x))
	demLine.set_ydata([np.nan] * len(x))
	gasLine.set_ydata([np.nan] * len(x))
	actLine.set_ydata([np.nan] * len(x))
	rewLine.set_ydata([np.nan] * len(x))
	return gasLine, actLine, demLine, prodLine

def animate(i):
    prodLine.set_ydata(production[i:breadth+i])
    demLine.set_ydata(demand[i:breadth+i])
    gasLine.set_ydata(gasProd[i:breadth+i])
    actLine.set_ydata(avgActions[i:breadth+i])
    rewLine.set_ydata(avgRewards[i:breadth+i])
    return gasLine, actLine, demLine, prodLine

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, blit=True, save_count=50)

plt.show()