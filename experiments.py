import json
import numpy as np
import yfinance as yf
from coskweness_cokurtosis import coskewness, cokurtosis
from portfolio_hubo_qaoa import HigherOrderPortfolioQAOA
import os

experiments = None
with open("experiments_data.json", "r") as f:
    experiments = list(json.load(f)["data"])

output_file = "portfolio_optimization_results.json"

init_experiment = 0
# Load existing data if file exists
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        existing_results = json.load(f)
    init_experiment = max([int(k) for k in existing_results.keys()]) + 1


for i, experiment in enumerate(experiments[init_experiment:]):
    results_for_experiment = {}
    stocks = experiment["stocks"]
    start = experiment["start"]
    end = experiment["end"]
    risk_aversion = 3
    n_layers = 5
    max_qubits = 15
    budget = experiment["budget"]
    print(budget)

    data = yf.download(stocks, start=start, end=end)
    prices_now = data["Close"].iloc[-1]
        
    returns = data["Close"].pct_change().dropna()
    stocks = returns.columns

    numpy_returns = returns.to_numpy()
    expected_returns = numpy_returns.mean(axis=0)*252
    covariance_matrix = np.cov(numpy_returns, rowvar=False)*252
    coskewness_tensor = coskewness(numpy_returns)#*(252**2)
    cokurtosis_tensor = cokurtosis(numpy_returns)#*(252**3)

    portfolio_hubo = HigherOrderPortfolioQAOA(stocks=stocks,
                                            prices_now=prices_now,
                                            expected_returns=expected_returns, 
                                            covariance_matrix=covariance_matrix,
                                            budget=budget,
                                            max_qubits = max_qubits,
                                            coskewness_tensor=coskewness_tensor, 
                                            cokurtosis_tensor=cokurtosis_tensor,
                                            log_encoding = True, 
                                            layers = n_layers,
                                            risk_aversion = risk_aversion)
    
    assets_to_qubits = portfolio_hubo.get_assets_to_qubits()
    
    weights, allocation, value, left_overs = portfolio_hubo.solve_with_continuous_variables()

    continuous_variables_solution = {
        "weights": weights,
        "allocation": allocation,
        "value": value,
        "left_overs": left_overs
    }

    (
        smallest_eigenvalues, 
        smallest_bitstrings, 
        first_excited_energy, 
        optimized_portfolio, 
        second_optimized_portfolio,
        eigenvalues,
        result1, 
        result2
    ) = portfolio_hubo.solve_exactly()

    exact_solution = {
        "smallest_eigenvalues": smallest_eigenvalues,
        "smallest_bitstrings": ["".join([str(i) for i in bits]) for bits in smallest_bitstrings],
        "first_excited_energy": first_excited_energy,
        "optimized_portfolio": optimized_portfolio,
        "second_optimized_portfolio": second_optimized_portfolio,
        "spectrum": eigenvalues,
        "result_with_budget": result1,
        "result_with_budget_excited": result2
    }

    for key, value in exact_solution.items():
        if key != "spectrum":
            print(f"{key}: {value}")
    
    (
        two_most_probable_states, 
        final_expectation_value, 
        params, 
        total_steps, 
        states_probs, 
        optimized_portfolios,
        training_history,
        objective_values,
        result1
    ) = portfolio_hubo.solve_with_qaoa_cma_es() #solve_with_qaoa_jax()

    qaoa_solution = {
        "two_most_probable_states": two_most_probable_states,
        "final_expectation_value": float(final_expectation_value),
        "params": params.tolist(),
        "total_steps": total_steps,
        "states_probs": [float(v) for v in states_probs],
        "optimized_portfolios": optimized_portfolios,
        "training_history": training_history,
        "objective_values": objective_values,
        "result_with_budget": result1
    }

    for key, value in qaoa_solution.items():
        if key != "training_history":
            print(f"{key}: {value}")

    hyperparams = {
        "stocks": [str(s) for s in stocks],
        "start": start,
        "end": end,
        "risk_aversion": risk_aversion,
        "max_qubits": max_qubits,
        "budget": budget,
        "log_encoding": True,
        "layers": n_layers,
        "prices_now": {str(k): float(v) for k, v in prices_now.items()},
        "assets_to_qubits": {str(k): v for k, v in assets_to_qubits.items()}
    }

    results_for_experiment["hyperparams"] = hyperparams
    results_for_experiment["continuous_variables_solution"] = continuous_variables_solution
    results_for_experiment["exact_solution"] = exact_solution
    results_for_experiment["qaoa_solution"] = qaoa_solution

    # Load existing data if file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    existing_results[i + init_experiment] = results_for_experiment

    # Write the updated results back to the file
    with open(output_file, 'w') as f:
        json.dump(existing_results, f, indent=4)