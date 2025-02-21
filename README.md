# Genetic Algorithm for PID Optimization

## Description
This project implements a **Genetic Algorithm (GA)** to optimize **PID parameters** in a dynamic system. The program evaluates the PID controller's performance based on overshoot, steady-state error, and settling time.

## Installation
To run the code, install the following Python packages:
```bash
pip install numpy matplotlib control
```

## Usage
Run the main program by executing:
```bash
python PID-GA2.py
```
Adjustable parameters:
- `populationSize`: Number of candidates in the population
- `generations`: Number of iterations the GA will run
- `mutationRate`: Probability of mutation for a PID parameter

## Modifying the System's Transfer Function
You can change the system's transfer function by adjusting `num` (numerator) and `den` (denominator) in the `system()` function in the code. This allows you to simulate different dynamic systems for GA-based optimization.

## Example Execution
When the program runs, it will print the best PID parameters per generation and finally plot the step response for the optimized system.

Example output:
```
Generation 99: Best fitness: 0.245, Best PID: (P=2.1, I=0.8, D=0.05)
```

## Theoretical Background
### PID Control
A PID controller adjusts the system's response based on proportional, integral, and derivative errors:
$$
G_c(s) = K_p + \frac{K_i}{s} + K_d s
$$

### Genetic Algorithm (GA)
GA is an evolution-based method that simulates natural selection to find optimal solutions.

## References
- PRACTICAL GENETIС ALGORITHMS by Randy L. Haupt and Sue Ellen Haupt
- ChatGPT-generated content for README structure and Markdown formatting.
- Some parts of the Genetic Algorithm implementation were inspired by ChatGPT (Feb 2025) (Coment where relevant)

## Author
This project was developed as part of the AIS2101 course.


