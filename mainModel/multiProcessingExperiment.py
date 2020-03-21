from markovSources import SolarPanel
from time import perf_counter, sleep
import concurrent.futures

start = perf_counter()

numPanels = 100
solarPanels = []
solarOutputs = []
totalOutput = 0
for i in range(0,numPanels):
	solarOutputs.append(0)
	solarPanels.append(SolarPanel())

# for panel in solarPanels:
# 	totalOutput += panel.update(0)

# with futures.ProcessPoolExecutor as executor:
# 	results = [executor.sumbit(panel.update[0]) for panel in solarPanels]

# 	for output in futures.as_completed(results):
# 		print(output)

def do_something(secs):
	sleep(secs)
	return secs

with concurrent.futures.ProcessPoolExecutor as executor:
	secs = [5,4,3,2,1]
	results = [executor.sumbit(do_something, sec) for sec in seconds]

	# for f in concurrent.futures.as_completed(results):
	# 	print(f.result())

print(totalOutput)

end = perf_counter()
print(end - start)