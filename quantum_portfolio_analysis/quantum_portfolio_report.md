# Quantum Portfolio Optimization Analysis Report

## Basic Statistics
- Number of samples: 100
- Stocks used: NKE, TRV, CAT, IBM, DOW, GS, AMGN, MSFT, MCD, V, CVX, DIS, VZ, UNH, AXP, WBA, KO, HD, WMT, CRM, HON, JNJ, BA, MRK, INTC, AAPL, CSCO, JPM, MMM, PG
- Average budget: $2455.94
- Average qubit count: 10.5
- Average QAOA layers: 9.0

## Solution Quality Comparison
- Classical vs. Exact: 48.20%
- Classical vs. QAOA: -3142.55%
- Exact vs. QAOA: -17281.97%
- QAOA-Exact agreement rate: 4.0%

### Budget Utilization
- Classical: 95.7%
- Exact: 99.9%
- QAOA: 88.0%

## QAOA Performance Analysis
- Average expectation value: -0.227433
- Average convergence rate: 1.00
- Average optimization steps: 811.8

### QAOA Performance by Layer Count
- 6 layers: -0.343786
- 7 layers: -0.306755
- 8 layers: -0.342441
- 9 layers: -0.294798
- 10 layers: -0.164425

## Quantum Solution Bitstring Analysis
- Average Hamming weight: 3.51
- Solution diversity average: 5.97

## Classical vs Exact Performance
- Classical better: 39 times
- Exact better: 61 times
- QAOA better than classical: 21 times

### Most Common Solution Bitstrings
- 1110100: 2 occurrences
- 001000: 2 occurrences
- 0001010110: 1 occurrences
- 1000000011: 1 occurrences
- 0110001101000: 1 occurrences
