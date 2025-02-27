import json
import numpy as np
import yfinance as yf
from coskweness_cokurtosis import coskewness, cokurtosis
from portfolio_hubo_qaoa import HigherOrderPortfolioQAOA

experiments = None
with open("experiments_data.json", "r") as f:
    experiments = json.load(f)

results = {}

for experiment_id in experiments:
    for experiment in experiments[experiment_id]:
        results_for_experiment = {}
        print("Solving experiment")
        print(experiment)
        stocks = experiment["stocks"]
        if len(stocks) < 4:
            print("Not enough stocks")
            continue
        start = experiment["start"]
        end = experiment["end"]
        risk_aversion = 3

        data = yf.download(stocks, start=start, end=end)
        prices_now = data["Close"].iloc[-1]
        print(prices_now)

        max_qubits = 15
        cheapest_stock = prices_now.mean()
        budget = 4*cheapest_stock
        print("Budget: ", budget)
        
        returns = data["Close"].pct_change().dropna()
        stocks = returns.columns

        numpy_returns = returns.to_numpy()
        expected_returns = numpy_returns.mean(axis=0)*252
        #print(returns.mean())
        #print(expected_returns)
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
                                                layers = 5,
                                                risk_aversion = risk_aversion)

        (
            smallest_eigenvalues, 
            smallest_bitstrings, 
            first_excited_energy, 
            optimized_portfolio, 
            second_optimized_portfolio 
        ) = portfolio_hubo.solve_exactly()

        exact_solution = {
            "smallest_eigenvalues": smallest_eigenvalues,
            "smallest_bitstrings": smallest_bitstrings,
            "first_excited_energy": first_excited_energy,
            "optimized_portfolio": optimized_portfolio,
            "second_optimized_portfolio": second_optimized_portfolio
        }

        for key, value in exact_solution.items():
            print(f"{key}: {value}")
        
        (
            two_most_probable_states, 
            final_expectation_value, 
            params, 
            total_steps, 
            states_probs, 
            optimized_portfolios
        ) = portfolio_hubo.solve_with_qaoa_jax()

        qaoa_solution = {
            "two_most_probable_states": [float(v) for v in two_most_probable_states],
            "final_expectation_value": float(final_expectation_value),
            "params": params.tolist(),
            "total_steps": total_steps,
            "states_probs": [float(v) for v in states_probs],
            "optimized_portfolios": optimized_portfolios
        }

        for key, value in qaoa_solution.items():
            print(f"{key}: {value}")

        results_for_experiment["exact_solution"] = exact_solution
        results_for_experiment["qaoa_solution"] = qaoa_solution
        break

    results[experiment_id] = results_for_experiment
    break

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)