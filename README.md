# Genetic Algorithm for PID optimization

## Description:
This project implements a **Genetic Algorithm (GA)** to try to optimize **PID-parameters** in a dynamic system. The programe evaluates the PID-parameters based on overshoot, settling time and steady state error during a step response.

## Dependencies:
To run the code the following libraries are required:

```bash
pip install numpy matplotlib control
```

## How to use:

Kjør hovedprogrammet ved å kjøre:
```bash
python PID-GA2.py
```
Parametere som kan justeres:
- `populationSize`: Antall kandidater i populasjonen
- `generations`: Antall iterasjoner GA skal kjøre
- `mutationRate`: Sannsynlighet for mutasjon av en PID-parameter

## Eksempelkjøring
Når programmet kjøres, vil det printe beste PID-parametere per generasjon, og til slutt plotte stegresponsen for det optimale systemet.

Eksempeloutput:
```
Generation 99: Best fitness: 0.245, Best PID: (P=2.1, I=0.8, D=0.05)
```

## Teoretisk bakgrunn
### PID-regulering
En PID-kontroller justerer systemets respons basert på proporsjonal, integrert og deriverte feil:
$$
G_c(s) = K_p + rac{K_i}{s} + K_d s
$$

### Genetisk Algoritme (GA)
GA er en evolusjonsbasert metode som simulerer naturlig seleksjon for å finne optimale løsninger.

## Referanser
- K. Åström & T. Hägglund, *PID Controllers: Theory, Design, and Tuning*, 2nd Edition.
- Forelesningsnotater fra kurset AIS2101
- Wikipedia: [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)

## Forfatter
Dette prosjektet ble utviklet som en del av AIS2101-kurset.
