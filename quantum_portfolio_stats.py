import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import math
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class QuantumPortfolioStats:
    """Analyze quantum portfolio optimization results across multiple files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize with directory containing portfolio optimization JSON files.
        
        Args:
            data_dir: Path to directory containing JSON result files
        """
        self.data_dir = data_dir
        self.results = []
        self.stats = {}
        
    def load_data(self) -> None:
        """Load all JSON files from the data directory."""
        json_files = list(Path(self.data_dir).glob('*.json'))
        print(f"Found {len(json_files)} JSON files to analyze")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Assuming data structure has numeric keys as the first level
                    for key, value in data.items():
                        if isinstance(value, dict) and all(
                            k in value for k in ['hyperparams', 'continuous_variables_solution', 
                                                'exact_solution', 'qaoa_solution']):
                            self.results.append(value)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Successfully loaded {len(self.results)} portfolio optimization results")
    
    def compute_basic_stats(self) -> Dict:
        """Compute basic statistics across all results."""
        if not self.results:
            print("No data loaded. Please run load_data() first.")
            return {}
        
        stats = {
            "num_samples": len(self.results),
            "stocks_used": set(),
            "risk_aversion_values": [],
            "qubit_counts": [],
            "qaoa_layer_counts": [],
            "average_budget": 0,
            "average_leftover_classical": 0,
        }
        
        total_budget = 0
        total_leftover = 0
        
        for result in self.results:
            # Gather stats about hyperparameters
            hyperparams = result["hyperparams"]
            stats["stocks_used"].update(hyperparams["stocks"])
            stats["risk_aversion_values"].append(hyperparams["risk_aversion"])
            stats["qubit_counts"].append(hyperparams["n_qubits"])
            stats["qaoa_layer_counts"].append(hyperparams["layers"])
            total_budget += hyperparams["budget"]
            
            # Leftover budget in classical solution
            total_leftover += result["continuous_variables_solution"]["left_overs"]
        
        stats["average_budget"] = total_budget / len(self.results)
        stats["average_leftover_classical"] = total_leftover / len(self.results)
        stats["stocks_used"] = list(stats["stocks_used"])
        
        self.stats["basic"] = stats
        return stats
    
    def compare_solution_quality(self) -> Dict:
        """Compare quality of classical, exact quantum, and QAOA solutions."""
        if not self.results:
            print("No data loaded. Please run load_data() first.")
            return {}
        
        stats = {
            "classical_vs_exact": [],  # Percentage improvement of exact over classical
            "classical_vs_qaoa": [],   # Percentage improvement of QAOA over classical
            "exact_vs_qaoa": [],       # Percentage difference between exact and QAOA
            "qaoa_probability_highest": [],  # Probability of most likely state in QAOA
            "qaoa_vs_exact_agreement": [],   # Whether QAOA found the exact ground state
            "budget_utilization": [],   # How well each method used the available budget
        }
        
        for result in self.results:
            # Extract objective values
            penalty = 10
            classical_obj = result["continuous_variables_solution"]["value"] - penalty*np.abs(result["continuous_variables_solution"]["left_overs"])
            
            # For exact solution, use the result with budget if available
            if "result_with_budget" in result["exact_solution"] and result["exact_solution"]["result_with_budget"]:
                exact_obj = result["exact_solution"]["result_with_budget"][-1]["objective_value"] - penalty*np.abs(result["exact_solution"]["result_with_budget"][-1]["difference"])
            #else:
                # Use the smallest eigenvalue if objective value not explicitly provided
            #    exact_obj = result["exact_solution"]["smallest_eigenvalues"][0] * -1  # Assuming minimization problem
            
            # For QAOA solution
            qaoa_expectation = result["qaoa_solution"]["final_expectation_value"]
            if "objective_values" in result["qaoa_solution"] and result["qaoa_solution"]["objective_values"]:
                qaoa_obj = result["qaoa_solution"]["objective_values"][-1] - penalty*np.abs(result["qaoa_solution"]["result_with_budget"][-1]["difference"])  # Take best objective value
            #else:
            #    qaoa_obj = qaoa_expectation * -1  # Approximate if not explicitly given
            
            # Calculate percentage improvements (taking care of sign based on minimization/maximization)
            # Assuming smaller (more negative) values are better
            if classical_obj != 0:
                classical_vs_exact = (exact_obj - classical_obj) / abs(classical_obj) * 100
                classical_vs_qaoa = (qaoa_obj - classical_obj) / abs(classical_obj) * 100
            else:
                classical_vs_exact = 0
                classical_vs_qaoa = 0
                
            if exact_obj != 0:
                exact_vs_qaoa = (qaoa_obj - exact_obj) / abs(exact_obj) * 100
            else:
                exact_vs_qaoa = 0
            
            stats["classical_vs_exact"].append(classical_vs_exact)
            stats["classical_vs_qaoa"].append(classical_vs_qaoa)
            stats["exact_vs_qaoa"].append(exact_vs_qaoa)
            
            # QAOA probability analysis
            if "states_probs" in result["qaoa_solution"] and result["qaoa_solution"]["states_probs"]:
                stats["qaoa_probability_highest"].append(result["qaoa_solution"]["states_probs"][0])
            else:
                stats["qaoa_probability_highest"].append(np.nan)
            
            # Check if QAOA found the exact ground state
            if "smallest_bitstrings" in result["exact_solution"] and "two_most_probable_states" in result["qaoa_solution"]:
                exact_bitstring = result["exact_solution"]["smallest_bitstrings"][0]
                qaoa_bitstrings = result["qaoa_solution"]["two_most_probable_states"]
                stats["qaoa_vs_exact_agreement"].append(exact_bitstring in qaoa_bitstrings)
            else:
                stats["qaoa_vs_exact_agreement"].append(False)
            
            # Budget utilization
            budget = result["hyperparams"]["budget"]
            
            # Classical solution budget usage
            classical_leftover = result["continuous_variables_solution"]["left_overs"]
            classical_util = (budget - classical_leftover) / budget
            
            # Exact solution budget usage (if available)
            if "result_with_budget" in result["exact_solution"] and result["exact_solution"]["result_with_budget"]:
                exact_leftover = result["exact_solution"]["result_with_budget"][-1]["difference"]
                exact_util = (budget - exact_leftover) / budget if exact_leftover < budget else 1.0
            else:
                exact_util = np.nan
                
            # QAOA solution budget usage (if available)
            if "result_with_budget" in result["qaoa_solution"] and result["qaoa_solution"]["result_with_budget"]:
                # Get the budget usage of the first solution
                qaoa_leftover = result["qaoa_solution"]["result_with_budget"][-1]["difference"]
                qaoa_util = (budget - qaoa_leftover) / budget if qaoa_leftover < budget else 1.0
            else:
                qaoa_util = np.nan
                
            stats["budget_utilization"].append({
                "classical": classical_util,
                "exact": exact_util,
                "qaoa": qaoa_util
            })
        
        # Calculate aggregated statistics
        stats["avg_classical_vs_exact"] = np.mean(stats["classical_vs_exact"])
        stats["avg_classical_vs_qaoa"] = np.mean(stats["classical_vs_qaoa"])
        stats["avg_exact_vs_qaoa"] = np.mean(stats["exact_vs_qaoa"])
        stats["avg_qaoa_probability"] = np.nanmean(stats["qaoa_probability_highest"])
        stats["qaoa_exact_agreement_rate"] = np.mean(stats["qaoa_vs_exact_agreement"])
        
        # Calculate average budget utilization
        stats["avg_budget_utilization"] = {
            "classical": np.mean([item["classical"] for item in stats["budget_utilization"]]),
            "exact": np.nanmean([item["exact"] for item in stats["budget_utilization"]]),
            "qaoa": np.nanmean([item["qaoa"] for item in stats["budget_utilization"]])
        }
        
        self.stats["solution_quality"] = stats
        return stats
    
    def analyze_qaoa_performance(self) -> Dict:
        """Analyze QAOA performance metrics across samples."""
        if not self.results:
            print("No data loaded. Please run load_data() first.")
            return {}
        
        stats = {
            "qaoa_steps": [],
            "qaoa_expectation_values": [],
            "qaoa_param_count": [],
            "qaoa_convergence": [],  # How many steps to reach 95% of final value
            "qaoa_vs_layers": []     # Relationship between performance and layer count
        }
        
        for result in self.results:
            qaoa = result["qaoa_solution"]
            layers = result["hyperparams"]["layers"]
            
            # Basic QAOA metrics
            stats["qaoa_expectation_values"].append(qaoa["final_expectation_value"])
            stats["qaoa_param_count"].append(len(qaoa["params"]) if "params" in qaoa else 0)
            
            # Steps information
            if "total_steps" in qaoa:
                stats["qaoa_steps"].append(qaoa["total_steps"])
            
            # QAOA performance vs layer relationship
            stats["qaoa_vs_layers"].append({
                "layers": layers,
                "expectation": qaoa["final_expectation_value"]
            })
            
            # Convergence analysis (if training history is available)
            if "training_history" in qaoa and "fbest" in qaoa["training_history"]:
                final_value = qaoa["training_history"]["fbest"]
                evals_best = qaoa["training_history"]["evals_best"]
                total_evaluations = qaoa["training_history"]["evaluations"]
                
                # Calculate convergence rate (percentage of evaluations needed to reach best)
                convergence_rate = evals_best / total_evaluations if total_evaluations > 0 else 1.0
                stats["qaoa_convergence"].append(convergence_rate)
        
        # Calculate aggregated statistics
        stats["avg_qaoa_steps"] = np.mean(stats["qaoa_steps"]) if stats["qaoa_steps"] else np.nan
        stats["avg_qaoa_expectation"] = np.mean(stats["qaoa_expectation_values"])
        stats["avg_qaoa_param_count"] = np.mean(stats["qaoa_param_count"])
        stats["avg_qaoa_convergence"] = np.mean(stats["qaoa_convergence"]) if stats["qaoa_convergence"] else np.nan
        
        # Analyze QAOA performance by layer count
        layer_performance = defaultdict(list)
        for item in stats["qaoa_vs_layers"]:
            layer_performance[item["layers"]].append(item["expectation"])
        
        stats["performance_by_layers"] = {
            layers: np.mean(values) for layers, values in layer_performance.items()
        }
        
        self.stats["qaoa_performance"] = stats
        return stats
    
    def analyze_bitstring_distributions(self) -> Dict:
        """Analyze the distribution of bitstrings in quantum solutions."""
        if not self.results:
            print("No data loaded. Please run load_data() first.")
            return {}
        
        stats = {
            "bitstring_patterns": defaultdict(int),
            "hamming_weights": [],  # Number of 1s in solution bitstrings
            "solution_diversity": []  # How different are the top solutions
        }
        
        for result in self.results:
            # Extract bitstrings from exact solution
            if "smallest_bitstrings" in result["exact_solution"]:
                for bitstring in result["exact_solution"]["smallest_bitstrings"]:
                    stats["bitstring_patterns"][bitstring] += 1
                    # Calculate Hamming weight (number of 1s)
                    stats["hamming_weights"].append(bitstring.count('1'))
            
            # Solution diversity: check how different are the top solutions
            # Compare exact solution with second best
            if ("smallest_bitstrings" in result["exact_solution"] and 
                "second_optimized_portfolio" in result["exact_solution"] and result["exact_solution"]["second_optimized_portfolio"] is not None):
                
                exact_portfolio = result["exact_solution"]["optimized_portfolio"][0]
                second_portfolio = result["exact_solution"]["second_optimized_portfolio"][0]
                
                # Calculate diversity as sum of absolute differences in allocations
                diversity = sum(abs(exact_portfolio.get(stock, 0) - second_portfolio.get(stock, 0)) 
                               for stock in set(exact_portfolio) | set(second_portfolio))
                
                stats["solution_diversity"].append(diversity)
        
        # Convert bitstring patterns to sorted list of (bitstring, count) tuples
        stats["bitstring_patterns"] = sorted(
            [(k, v) for k, v in stats["bitstring_patterns"].items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate statistics about Hamming weights
        if stats["hamming_weights"]:
            stats["avg_hamming_weight"] = np.mean(stats["hamming_weights"])
            stats["min_hamming_weight"] = np.min(stats["hamming_weights"])
            stats["max_hamming_weight"] = np.max(stats["hamming_weights"])
        
        # Calculate statistics about solution diversity
        if stats["solution_diversity"]:
            stats["avg_solution_diversity"] = np.mean(stats["solution_diversity"])
            stats["max_solution_diversity"] = np.max(stats["solution_diversity"])
            stats["min_solution_diversity"] = np.min(stats["solution_diversity"])
        
        self.stats["bitstring_analysis"] = stats
        return stats
    
    def compute_how_many_times_classical_outperformed_exact(self) -> Dict:
        """Compute how many times QAOA outperformed the exact solution."""
        if not self.results:
            print("No data loaded. Please run load_data() first.")
            return {}
        
        classical_best_count = 0
        exact_best_count = 0
        qaoa_better_than_classical_count = 0
        
        for result in self.results:
            classical_obj = result["continuous_variables_solution"]["value"] - np.abs(result["continuous_variables_solution"]["left_overs"])
            exact_obj = result["exact_solution"]["result_with_budget"][-1]["objective_value"] - np.abs(result["exact_solution"]["result_with_budget"][-1]["difference"])
            qaoa_obj = result["qaoa_solution"]["objective_values"][-1] - np.abs(result["qaoa_solution"]["result_with_budget"][-1]["difference"])
            
            if classical_obj > exact_obj:
                classical_best_count += 1
            else:
                exact_best_count += 1
            
            if qaoa_obj > classical_obj:
                qaoa_better_than_classical_count += 1
        
        return {
            "exact_better_count": exact_best_count,
            "classical_best_count": classical_best_count,
            "qaoa_better_than_classical_count": qaoa_better_than_classical_count
        }
    
    def compute_all_stats(self) -> Dict:
        """Compute all statistics."""
        self.compute_basic_stats()
        self.compare_solution_quality()
        self.analyze_qaoa_performance()
        self.analyze_bitstring_distributions()
        #self.compute_how_many_times_classical_outperformed_exact()
        return self.stats
    
    def visualize_results(self, output_dir: str = None) -> None:
        """Create visualizations of the statistical results."""
        if not self.stats:
            print("No statistics computed. Please run compute_all_stats() first.")
            return
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set the style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Solution Quality Comparison
        self._plot_solution_quality_comparison(output_dir)
        
        # 2. QAOA Performance by Layer Count
        self._plot_qaoa_by_layers(output_dir)
        
        # 3. Budget Utilization Comparison
        self._plot_budget_utilization(output_dir)
        
        # 4. Bitstring Hamming Weight Distribution
        self._plot_hamming_weight_distribution(output_dir)
        
        # 5. QAOA Convergence Plot
        self._plot_qaoa_convergence(output_dir)
        
        # 6. Solution Diversity
        self._plot_solution_diversity(output_dir)
    
    def _plot_solution_quality_comparison(self, output_dir: str = None) -> None:
        """Plot comparison of solution quality between methods."""
        if "solution_quality" not in self.stats:
            return
            
        quality_stats = self.stats["solution_quality"]
        
        plt.figure(figsize=(10, 6))
        comparisons = [
            quality_stats["avg_classical_vs_exact"],
            quality_stats["avg_classical_vs_qaoa"],
            quality_stats["avg_exact_vs_qaoa"]
        ]
        labels = ['Classical vs. Exact', 'Classical vs. QAOA', 'Exact vs. QAOA']
        
        plt.bar(labels, comparisons, color=['#2c7bb6', '#d7191c', '#fdae61'])
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylabel('% Improvement')
        plt.title('Solution Quality Comparison Between Methods')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(comparisons):
            plt.text(i, v + (5 if v >= 0 else -5), 
                    f'{v:.2f}%', 
                    ha='center', va='center' if v >= 0 else 'top',
                    fontweight='bold')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'solution_quality_comparison.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_qaoa_by_layers(self, output_dir: str = None) -> None:
        """Plot QAOA performance by layer count."""
        if "qaoa_performance" not in self.stats:
            return
            
        qaoa_stats = self.stats["qaoa_performance"]
        
        if not qaoa_stats["performance_by_layers"]:
            return
            
        plt.figure(figsize=(10, 6))
        
        layers = sorted(qaoa_stats["performance_by_layers"].keys())
        performances = [qaoa_stats["performance_by_layers"][l] for l in layers]
        
        plt.plot(layers, performances, 'o-', linewidth=2, markersize=10, color='#5ab4ac')
        plt.xlabel('Number of QAOA Layers')
        plt.ylabel('Average Expectation Value')
        plt.title('QAOA Performance by Layer Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(performances):
            plt.text(layers[i], v, f'{v:.4f}', ha='left', va='bottom')
        
        # Set integer ticks for x-axis
        plt.xticks(layers)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'qaoa_performance_by_layers.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_budget_utilization(self, output_dir: str = None) -> None:
        """Plot budget utilization comparison."""
        if "solution_quality" not in self.stats:
            return
            
        budget_stats = self.stats["solution_quality"]["avg_budget_utilization"]
        
        plt.figure(figsize=(10, 6))
        
        methods = list(budget_stats.keys())
        utilization = [budget_stats[m] * 100 for m in methods]  # Convert to percentage
        
        plt.bar(methods, utilization, color=['#1b9e77', '#d95f02', '#7570b3'])
        plt.ylabel('Budget Utilization (%)')
        plt.title('Average Budget Utilization by Method')
        plt.ylim(0, 105)  # Leave some room for labels
        
        # Add value labels on bars
        for i, v in enumerate(utilization):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'budget_utilization.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_hamming_weight_distribution(self, output_dir: str = None) -> None:
        """Plot distribution of Hamming weights (number of 1s) in solution bitstrings."""
        if "bitstring_analysis" not in self.stats:
            return
            
        bitstring_stats = self.stats["bitstring_analysis"]
        
        if not bitstring_stats["hamming_weights"]:
            return
            
        plt.figure(figsize=(10, 6))
        
        weights = bitstring_stats["hamming_weights"]
        
        # Create histogram
        plt.hist(weights, bins=range(min(weights), max(weights) + 2), 
                 edgecolor='black', alpha=0.7, color='#8da0cb')
        
        plt.xlabel('Hamming Weight (Number of 1s)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Hamming Weights in Solution Bitstrings')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add statistics as text
        stats_text = (f"Mean: {bitstring_stats['avg_hamming_weight']:.2f}\n"
                      f"Min: {bitstring_stats['min_hamming_weight']}\n"
                      f"Max: {bitstring_stats['max_hamming_weight']}")
        
        plt.figtext(0.15, 0.8, stats_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.5))
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'hamming_weight_distribution.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_qaoa_convergence(self, output_dir: str = None) -> None:
        """Plot QAOA convergence statistics."""
        if "qaoa_performance" not in self.stats:
            return
            
        qaoa_stats = self.stats["qaoa_performance"]
        
        if not qaoa_stats["qaoa_convergence"]:
            return
            
        plt.figure(figsize=(10, 6))
        
        convergence_rates = qaoa_stats["qaoa_convergence"]
        
        # Create histogram of convergence rates
        plt.hist(convergence_rates, bins=10, edgecolor='black', alpha=0.7, color='#fc8d62')
        
        plt.xlabel('Convergence Rate (Fraction of Evaluations to Reach Best)')
        plt.ylabel('Frequency')
        plt.title('QAOA Convergence Rate Distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add average as vertical line
        avg_convergence = qaoa_stats["avg_qaoa_convergence"]
        plt.axvline(x=avg_convergence, color='red', linestyle='--', linewidth=2)
        plt.text(avg_convergence + 0.01, plt.ylim()[1]*0.9, 
                f'Average: {avg_convergence:.2f}', 
                va='center', color='red', fontweight='bold')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'qaoa_convergence.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_solution_diversity(self, output_dir: str = None) -> None:
        """Plot solution diversity statistics."""
        if "bitstring_analysis" not in self.stats:
            return
            
        diversity_stats = self.stats["bitstring_analysis"]
        
        if not diversity_stats["solution_diversity"]:
            return
            
        plt.figure(figsize=(10, 6))
        
        diversity_values = diversity_stats["solution_diversity"]
        
        # Create histogram of diversity values
        plt.hist(diversity_values, bins=10, edgecolor='black', alpha=0.7, color='#66c2a5')
        
        plt.xlabel('Solution Diversity (Sum of Absolute Differences)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Solution Diversity Between Top Solutions')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add statistics as text
        stats_text = (f"Mean: {diversity_stats['avg_solution_diversity']:.2f}\n"
                      f"Min: {diversity_stats['min_solution_diversity']}\n"
                      f"Max: {diversity_stats['max_solution_diversity']}")
        
        plt.figtext(0.15, 0.8, stats_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.5))
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'solution_diversity.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive textual report of the statistics."""
        if not self.stats:
            print("No statistics computed. Please run compute_all_stats() first.")
            return ""
        
        report = []
        report.append("# Quantum Portfolio Optimization Analysis Report\n")
        
        # Basic statistics
        report.append("## Basic Statistics")
        basic = self.stats.get("basic", {})
        report.append(f"- Number of samples: {basic.get('num_samples', 'N/A')}")
        report.append(f"- Stocks used: {', '.join(basic.get('stocks_used', []))}")
        report.append(f"- Average budget: ${basic.get('average_budget', 'N/A'):.2f}")
        report.append(f"- Average qubit count: {np.mean(basic.get('qubit_counts', [0])):.1f}")
        report.append(f"- Average QAOA layers: {np.mean(basic.get('qaoa_layer_counts', [0])):.1f}")
        report.append("")
        
        # Solution quality comparison
        report.append("## Solution Quality Comparison")
        quality = self.stats.get("solution_quality", {})
        report.append(f"- Classical vs. Exact: {quality.get('avg_classical_vs_exact', 'N/A'):.2f}%")
        report.append(f"- Classical vs. QAOA: {quality.get('avg_classical_vs_qaoa', 'N/A'):.2f}%")
        report.append(f"- Exact vs. QAOA: {quality.get('avg_exact_vs_qaoa', 'N/A'):.2f}%")
        report.append(f"- QAOA-Exact agreement rate: {quality.get('qaoa_exact_agreement_rate', 'N/A')*100:.1f}%")
        report.append("")
        
        # Budget utilization
        budget_util = quality.get("avg_budget_utilization", {})
        report.append("### Budget Utilization")
        report.append(f"- Classical: {budget_util.get('classical', 'N/A')*100:.1f}%")
        report.append(f"- Exact: {budget_util.get('exact', 'N/A')*100:.1f}%")
        report.append(f"- QAOA: {budget_util.get('qaoa', 'N/A')*100:.1f}%")
        report.append("")
        
        # QAOA Performance
        report.append("## QAOA Performance Analysis")
        qaoa = self.stats.get("qaoa_performance", {})
        report.append(f"- Average expectation value: {qaoa.get('avg_qaoa_expectation', 'N/A'):.6f}")
        report.append(f"- Average convergence rate: {qaoa.get('avg_qaoa_convergence', 'N/A'):.2f}")
        report.append(f"- Average optimization steps: {qaoa.get('avg_qaoa_steps', 'N/A'):.1f}")
        report.append("")
        
        # Performance by layers
        report.append("### QAOA Performance by Layer Count")
        perf_by_layers = qaoa.get("performance_by_layers", {})
        for layer, perf in sorted(perf_by_layers.items()):
            report.append(f"- {layer} layers: {perf:.6f}")
        report.append("")
        
        # Bitstring analysis
        report.append("## Quantum Solution Bitstring Analysis")
        bitstring = self.stats.get("bitstring_analysis", {})
        report.append(f"- Average Hamming weight: {bitstring.get('avg_hamming_weight', 'N/A'):.2f}")
        report.append(f"- Solution diversity average: {bitstring.get('avg_solution_diversity', 'N/A'):.2f}")
        report.append("")
        
        # compute_how_many_times_classical_outperformed_exact
        report.append("## Classical vs Exact Performance")
        classical_vs_exact = self.compute_how_many_times_classical_outperformed_exact()
        report.append(f"- Classical better: {classical_vs_exact['classical_best_count']} times")
        report.append(f"- Exact better: {classical_vs_exact['exact_better_count']} times")
        report.append(f"- QAOA better than classical: {classical_vs_exact['qaoa_better_than_classical_count']} times")
        report.append("")
        
        # Most common bitstrings
        report.append("### Most Common Solution Bitstrings")
        for bitstring, count in bitstring.get("bitstring_patterns", [])[:5]:  # Top 5
            report.append(f"- {bitstring}: {count} occurrences")
        report.append("")
        
        # Join report with newlines
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text