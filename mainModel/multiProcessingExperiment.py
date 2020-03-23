from markovSources import SolarPanel
from time import perf_counter
from multiprocessing import Pool

# we will attempt to update a bunch of solar panels both in parallel and in sequence to see which is faster

numSolarPanels = 200

solarPanels = []
for i in range(0, numSolarPanels):
	solarPanels.append(SolarPanel())

# in sequence:

no_mp_start = perf_counter()

total = 0
for panel in solarPanels:
	total += panel.update(1)

no_mp_end = perf_counter()
no_mp_time = no_mp_end - no_mp_start
print("time for one iteration without multiprocessing: "+str(no_mp_time))

# in parallel:

def do_thing(panel):
	return panel.update(1)

mp_start = perf_counter()

pool = Pool()
result = pool.map(do_thing, solarPanels)
pool.close()
pool.join()
sum(result)

mp_end = perf_counter()
mp_time = mp_end - mp_start
print("time for one iteration with multiprocessing: "+str(mp_time))