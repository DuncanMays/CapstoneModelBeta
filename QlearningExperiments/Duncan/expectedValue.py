from random import randint

expectedValue = 0
numIterations = 1000000
num = 1
for i in range(numIterations):
	sum = 0

	a = randint(-1, 1)
	b = randint(-1, 1)
	c = randint(-1, 1)

	sumand = a + b + c

	if (sumand > 0):
		alpha = 1/num

		expectedValue = alpha*sumand + (1-alpha)*expectedValue

		num += 1

print(expectedValue)