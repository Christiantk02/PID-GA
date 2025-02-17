import control as ctrl
import matplotlib.pyplot as plt
import numpy as np


def system():
    # Create a system transfer function G(s) = num/den
    num = [1,]
    den = [1, 2, 8]
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
    settlingTime = t[idx[-1]] - t[idx[0]]
    #fortsett her!!!!

    # Steady state error
    steadyState = abs(y[-1] - 1)

    return overshoot, settlingTime, steadyState

sys = system()
kp, ki, kd = 0.004, 19, 15
t, y = simulatePid(kp, ki, kd, sys)
print(fitness(kp, ki, kd, sys))
plot(t, y)
