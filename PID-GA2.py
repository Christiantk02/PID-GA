import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def system():
    # Generate a transferfunction for the system G(s) = num/den
    num = [1]
    den = [1,2,1]
    sys = ctrl.TransferFunction(num,den)
    return sys

def pidController(kp, ki, kd):
    # Genereate a pid transferfunction Gc(s) = kp + ki/s + kds
    controller = ctrl.TransferFunction([kd, kp, ki], [1, 0])
    return controller

def simulatePid(kp, ki, kd, system):
    # Generates feedback function og pid and system
    pid = pidController(kp, ki, kd)
    loop = ctrl.feedback(pid * system)
    t, y = ctrl.step_response(loop)
    return t, y 

def plot(t, y):
    # Plots a step response
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Step response')
    plt.show()

def fitness(kp, ki, kd, system):
    # Find fitness based on step response
    t, y = simulatePid(kp, ki, kd, system)

    overshoot = (max(y) -1)*100 if max(y) > 1 else 0

    steadyState = abs(y[-1] - 1)

    settlingTime = t[-1]

    idx = np.where(abs(y-1) < 0.02)[0]
    if (len(idx) > 0):
        for i in range(len(idx)):
            if np.all(abs(y[idx[i]:] - 1) < 0.02):
                settlingTime = t[idx[i]]
                break

    return overshoot**overshoot + steadyState*3 + settlingTime*0.3


def initializePopulation(size):
    # Initialize population with random values
    population = []
    for i in range(size):
        kp = rnd.uniform(0, 5)
        ki = rnd.uniform(0, 3)
        kd = rnd.uniform(0, 1)
        population.append((kp, ki, kd))
    return population

def evaluatePopulation(population, system):
    # Evaluate population based on fitness
    fitnessValues = []
    for i in range(len(population)):
        kp, ki, kd = population[i]
        fitnessValues.append((fitness(kp, ki, kd, system), (kp, ki, kd)))
    fitnessValues.sort(key=lambda x: x[0])  # Made By ChatGPT  
    return fitnessValues

def selectParents(population, fitnessValues):
    # Select parents based on fitness
    parents = []
    for i in range(len(population)//3):
        parents.append(fitnessValues[i][1])
    return parents

def crossover(parents, size):
    # Crossover parents to make new children using Arithmetic metod
    newPopulation = []
    newPopulation.append(parents[0])

    while (len(newPopulation) < size):
        alpha = rnd.uniform(0,1)
        p1, p2 = rnd.sample(parents, 2)
        kp = alpha*p1[0] + (1-alpha)*p2[0] 
        ki = alpha*p1[1] + (1-alpha)*p2[1] 
        kd = alpha*p1[2] + (1-alpha)*p2[2] 
        newPopulation.append((kp, ki, kd))
    return newPopulation

def mutate(population, mutationRate):
    # Mutate population
    newPopulation = []
    newPopulation.append(population[0])

    for i in range(len(population)-1):
        individual = list(population[i])  
        if rnd.random() < mutationRate:
            individual[0] += rnd.uniform(-0.8, 0.8)  
        if rnd.random() < mutationRate:
            individual[1] += rnd.uniform(-0.4, 0.4)  
        if rnd.random() < mutationRate:
            individual[2] += rnd.uniform(-0.05, 0.05)  
        newPopulation.append(tuple(individual))  
    return newPopulation

def geneticAlgorithm(system, populationSize, generations, mutationRate):
    # Run genetic algorithm
    population = initializePopulation(populationSize)

    for i in range(generations):
        fitnessValues = evaluatePopulation(population, system)

        parents = selectParents(population, fitnessValues)

        newPopulation = crossover(parents, populationSize)

        population = mutate(newPopulation, mutationRate)

        bestFitness, bestPid = fitnessValues[0]
        print(f'Generation {i}: Best fitness: {bestFitness}, Best PID: {bestPid}')

    return bestPid

sys = system()
bestP, bestI, bestD = geneticAlgorithm(sys,100, 100, 0.150)

print(f'Best PID: P={bestP}, I={bestI}, D={bestD}')
plot(*simulatePid(bestP, bestI, bestD, sys))
        






 

