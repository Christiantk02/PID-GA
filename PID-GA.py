import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def system():
    # Create a system transfer function G(s) = num/den
    num = [1]
    den = [1, 2, 1]
    sys = ctrl.TransferFunction(num, den)
    return sys

def pidController(kp, ki, kd):
    # Pid transfer function Gc(s) = kp + ki/s + kd*s
    controller = ctrl.TransferFunction([kd, kp, ki], [1, 0])
    return controller

def simulatePid(kp, ki, kd, system):
    # simulate pid controller
    pid = pidController(kp, ki, kd)
    loop = ctrl.feedback(pid * system)
    t, y = ctrl.step_response(loop)
    return t, y

def plot(t, y):
    # Plot step response
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Step response')
    plt.show()

def fitness(kp, ki, kd, system):
    # Find fitness based on step response
    t, y = simulatePid(kp, ki, kd, system)

    # Overshoot
    overshoot = max(y) - 1 if max(y) > 1 else 0

    # Settling time (+-2%)
    idx = np.where(np.abs(y -1) < 0.02)[0]
    if len(idx) > 0:
        for i in range(len(idx)):
            if np.all(np.abs(y[idx[i]:] -1) < 0.02):
                settlingTime = t[idx[i]]
                break
        else:
            settlingTime = t[-1]
    else:
        settlingTime = t[-1]
                

    # Steady state error
    steadyState = abs(y[-1] - 1)

    return overshoot + 0.3*settlingTime + steadyState

def initializePopulation(size):
    # Initialize population with random values
    population = []
    for i in range(size):
        kp = rnd.uniform(0, 3)
        ki = rnd.uniform(0, 1)
        kd = rnd.uniform(0, 0.5)
        population.append((kp, ki, kd))
    return population

def evaluatePopulation(population, system):
    # Evaluate population based on fitness
    fitnessValues = []
    for i in range(len(population)):
        kp, ki, kd = population[i]
        fitnessValues.append((fitness(kp, ki, kd, system), (kp, ki, kd)))
    fitnessValues.sort(key=lambda x: x[0])  # ikke peiling  
    return fitnessValues

def selectParents(population, fitnessValues):
    # Select parents based on fitness
    parents = []
    for i in range(len(population)//3):
        parents.append(fitnessValues[i][1])
    return parents

def crossover(parents, size):
    # Crossover parents to create new population
    newPopulation = []
    newPopulation.append(parents[0])
    
    while len(newPopulation) < size:
        p1, p2 = rnd.sample(parents, 2)
        newPopulation.append((
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2,
            (p1[2] + p2[2]) / 2
        ))    
    return newPopulation

def mutate(population, mutationRate):
    # Mutate population
    for i in range(len(population)):
        if rnd.random() < mutationRate:
            population[i] = (
                population[i][0] + rnd.uniform(-1, 1),
                population[i][1] + rnd.uniform(-0.2, 0.2),
                population[i][2] + rnd.uniform(-0.05, 0.05)
            )
    return population

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
bestP, bestI, bestD = geneticAlgorithm(sys,100, 100, 0.15)

print(f'Best PID: P={bestP}, I={bestI}, D={bestD}')
plot(*simulatePid(bestP, bestI, bestD, sys))