import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def system():
    # Generate a transferfunction for the system G(s) = num/den
    num = [1]
    den = [1,1]
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
    t, y = simulatePid(kp, ki, kd, system)

    overshoot = max(y) -1 if max(y) > 1 else 0

    steadyState = abs(y[-1] - 1)

    

