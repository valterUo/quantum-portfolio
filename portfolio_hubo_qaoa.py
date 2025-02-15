import itertools
import pennylane as qml
from pennylane import numpy as np
import scipy

np.random.seed(0)
from pennylane.transforms import compile

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

from pypfopt import EfficientFrontier

def replace_h_rz_h_with_rx(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    new_operations = []
    i = 0
    while i < len(tape.operations):
        op = tape.operations[i]

        # Detect pattern: H . RZ . H
        if (
            i + 2 < len(tape.operations)
            and op.name == "Hadamard"
            and tape.operations[i + 1].name == "RZ"
            and tape.operations[i + 2].name == "Hadamard"
            and op.wires == tape.operations[i + 1].wires == tape.operations[i + 2].wires
        ):
            rz_angle = tape.operations[i + 1].parameters[0]
            rx_angle = rz_angle  # RX(angle) = H . RZ(angle) . H
            new_operations.append(qml.RX(rx_angle, wires=op.wires[0]))

            # Skip the next two gates since they are replaced
            i += 3
        else:
            new_operations.append(op)
            i += 1

    # Create new transformed tape
    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        return results[0]

    return [new_tape], null_postprocessing


def smallest_eigenpairs(A):
    """
    Return the smallest eigenvalues and eigenvectors of a matrix A
    Returns always at least two eigenvalues and eigenvectors, 
    even if the second solution is not optimal.
    The non-zero difference between the two smallest eigenvalues 
    can describe hardness of the optimization problem.
    """

    eigenvalues, eigenvectors = scipy.linalg.eig(A)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    idx = np.argsort(eigenvalues)
    smallest_eigenvalues = []
    smallest_eigenvectors = []

    smallest_eigenvalue = eigenvalues[idx[0]]
    smallest_eigenvalues.append(smallest_eigenvalue)
    smallest_eigenvectors.append(eigenvectors[:, idx[0]])

    first_excited_energy = None
    first_excited_state = None
    
    # Find all smallest eigenvalues and eigenvectors
    for i in range(1, len(eigenvalues)):
        if eigenvalues[idx[i]] == smallest_eigenvalue:
            smallest_eigenvalues.append(eigenvalues[idx[i]])
            smallest_eigenvectors.append(eigenvectors[:, idx[i]])
        else:
            first_excited_energy = eigenvalues[idx[i]]
            first_excited_state = eigenvectors[:, idx[i]]
            break  
    
    return smallest_eigenvalues, smallest_eigenvectors, first_excited_energy, first_excited_state


def bitstring_to_int(bit_string_sample):
    if type(bit_string_sample[0]) == str:
        bit_string_sample = np.array([int(i) for i in bit_string_sample])
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample)

def int_to_bitstring(int_sample, n_qubits):
    bits = np.array([int(i) for i in format(int_sample, f'0{n_qubits}b')])
    return "".join([str(i) for i in bits])


def basis_vector_to_bitstring(basis_vector):
    assert np.sum(basis_vector) == 1, "Input must be a basis vector"
    # I think the basis vector returned from scipy.linalg.eig is "turned around"
    index = np.argmax(basis_vector[::-1])
    num_qubits = max(int(np.log2(len(basis_vector))), 1)
    bitstring = format(index, f'0{num_qubits}b')
    #bitstring = np.array(list(np.binary_repr(index).zfill(num_qubits)))
    bitstring = [int(i) for i in bitstring]
    return bitstring

def bitstrings_to_optimized_portfolios(bitstrings, assets_to_qubits):
    """
    Given a bitstring, return the portfolio that corresponds to the bitstring with log encoding
    """
    portfolios = []
    for bitstring in bitstrings:
        portfolio = {}
        for asset, qubits in assets_to_qubits.items():
            bits = [bitstring[q] for q in qubits]
            portfolio[asset] = bitstring_to_int(bits)
        portfolios.append(portfolio)
    return portfolios

class HigherOrderPortfolioQAOA:

    def __init__(self, 
                 stocks, 
                 expected_returns, 
                 covariance_matrix, 
                 budget, 
                 coskewness_tensor = None, 
                 cokurtosis_tensor = None, 
                 risk_aversion = 3, 
                 layers = 1, 
                 mixer = "x", 
                 log_encoding = False):
        # Implementation assumes that stocks and other data are ordered to match
        self.stocks = stocks
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        
        self.budget = budget
        if log_encoding:
            # Smallest N such that 2^N > budget
            N = int(np.floor(np.log2(budget)))
            self.num_qubits_per_asset = N if N > self.budget else N + 1
        else:
            self.num_qubits_per_asset = budget
        
        print("Number of qubits per asset: ", self.num_qubits_per_asset)
        
        self.coskewness_tensor = coskewness_tensor
        self.cokurtosis_tensor = cokurtosis_tensor
        self.risk_aversion = risk_aversion
        self.layers = layers
        self.mixer = mixer
        self.log_encoding = log_encoding

        self.num_assets = len(expected_returns)
        assert len(stocks) == len(expected_returns)
        assert len(expected_returns) == len(covariance_matrix)
        assert len(expected_returns) == len(covariance_matrix[0])
        assert risk_aversion > 0

        self.cost_hubo_int = {}
        self.cost_hubo_bin = {}
        self.assets_to_qubits = {}
        self.qubits_to_assets = {}

        self.total_qubits = 0
        for asset in stocks:
            self.assets_to_qubits[asset] = list(range(self.total_qubits, self.total_qubits + self.num_qubits_per_asset))
            for qubit in self.assets_to_qubits[asset]:
                self.qubits_to_assets[qubit] = asset
            self.total_qubits += self.num_qubits_per_asset

        print("Constructing cost hubo with integer variables")
        max_value = self.construct_cost_hubo_int()
        #print("Max value of the cost function: ", max_value)
        #print("Constructing budget constraint")

        ef = EfficientFrontier(self.expected_returns, self.covariance_matrix)
        weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)

        print("Optimized Weights (considering variance and returns):")
        for asset, weight in weights.items():
            print(f"{asset}: {weight:.2%}")

        numpy_weights = np.array(list(weights.values()))

        # Get utility with optimized weights
        maximized_utility = -0.5 * risk_aversion * numpy_weights.T @ self.covariance_matrix @ numpy_weights \
                            + self.expected_returns @ numpy_weights
        print("Maximized utility from classical mean variance: ", maximized_utility)
        scaler = max(maximized_utility, 1)
                
        self.budget_constraint = self.construct_budget_constraint(scaler=scaler)
        print("Adding budget constraints to the cost function -> constructing full hubo problem")
        for var, coeff in self.budget_constraint.items():
            if var in self.cost_hubo_int:
                self.cost_hubo_int[var] += coeff
            else:
                self.cost_hubo_int[var] = coeff
        print("Replacing integer variables with binary variables")
        self.replace_integer_variables_with_binary_variables()
        print("Simplifying the binary cost function")
        self.simplify_cost_hubo_bin()
        print("Converting binary cost function to Ising Hamiltonian")
        self.cost_hubo_bin_to_ising_hamiltonian()

        self.n_qubits = self.total_qubits
        print("Constructing QAOA circuits")
        self.qaoa_circuit, self.qaoa_probs_circuit = self.get_QAOA_circuits()
        self.init_params = 0.01*np.random.rand(2, self.layers, requires_grad=True)

    def construct_cost_hubo_int(self):
        max_value = 0
        for i in range(self.num_assets):
            mu = self.expected_returns[i]
            self.cost_hubo_int[(self.stocks[i],)] = -mu
            max_value += np.abs(mu)
        
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                cov = (self.risk_aversion/2)*self.covariance_matrix[i][j]
                self.cost_hubo_int[(self.stocks[i], self.stocks[j])] = cov
                max_value += np.abs(cov)

        if self.coskewness_tensor is not None:
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    for k in range(self.num_assets):
                        skew = (self.risk_aversion/6)*self.coskewness_tensor[i][j][k]
                        #self.cost_hubo_int[(self.stocks[i], self.stocks[j], self.stocks[k])] = -skew
                        max_value += np.abs(skew)
        
        if self.cokurtosis_tensor is not None:
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    for k in range(self.num_assets):
                        for l in range(self.num_assets):
                            kurt = (self.risk_aversion/24)*self.cokurtosis_tensor[i][j][k][l]
                            #self.cost_hubo_int[(self.stocks[i], self.stocks[j], self.stocks[k], self.stocks[l])] = kurt
                            max_value += np.abs(kurt)
        return max_value

    def construct_budget_constraint(self, scaler = 1):
        # Sum over all variables sums to budget (sum_{v in vars} v - budget)**2
        # Recall that the variables at this stage are integers so x^n != x
        budget_const = {}
        for asset in self.stocks:
            budget_const[(asset,)] = -2*self.budget*scaler
            budget_const[(asset, asset)] = scaler
        for asset1, asset2 in itertools.combinations(self.stocks, 2):
            budget_const[(asset1, asset2)] = 2*scaler
        return budget_const

    def replace_integer_variables_with_binary_variables(self):
        """
        Every integer variable m, labeled with stocks, is replaced with binary variables
        meaning integer variable m is replaced with a sum of binary variables x_0 + 2*x_1 + 4*x_2 + 8*x_3 + ... + 2^(n)*x_{n}
        where n = ceil(log2(budget)) and x_n are binary.
        Then, for integer variables m and k, we have m*k = sum_{i=0}^{n_m} sum_{j=0}^{n_k} 2^{i+j} x_i x_j, etc.
        """
        for integer_var in self.cost_hubo_int:
            if len(integer_var) == 1:
                asset = integer_var[0]
                for n, qubit in enumerate(self.assets_to_qubits[asset]):
                    bin_var = ((asset, qubit),)
                    if self.log_encoding:
                        if bin_var in self.cost_hubo_bin:
                            self.cost_hubo_bin[bin_var] += (2**n)*self.cost_hubo_int[integer_var]
                        else:
                            self.cost_hubo_bin[bin_var] = (2**n)*self.cost_hubo_int[integer_var]
                    else:
                        self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]
            elif len(integer_var) == 2:
                asset0 = integer_var[0]
                asset1 = integer_var[1]
                for n, qubit0 in enumerate(self.assets_to_qubits[asset0]):
                    for m, qubit1 in enumerate(self.assets_to_qubits[asset1]):
                        bin_var = ((asset0, qubit0), (asset1, qubit1))
                        if self.log_encoding:
                            if bin_var in self.cost_hubo_bin:
                                self.cost_hubo_bin[bin_var] += 2**(n+m)*self.cost_hubo_int[integer_var]
                            else:
                                self.cost_hubo_bin[bin_var] = 2**(n+m)*self.cost_hubo_int[integer_var]
                        else:
                            self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]
            elif len(integer_var) == 3:
                asset0 = integer_var[0]
                asset1 = integer_var[1]
                asset2 = integer_var[2]
                for n, qubit0 in enumerate(self.assets_to_qubits[asset0]):
                    for m, qubit1 in enumerate(self.assets_to_qubits[asset1]):
                        for k, qubit2 in enumerate(self.assets_to_qubits[asset2]):
                            bin_var = ((asset0, qubit0), (asset1, qubit1), (asset2, qubit2))
                            if self.log_encoding:
                                self.cost_hubo_bin[bin_var] = 2**(n+m+k)*self.cost_hubo_int[integer_var]
                            else:
                                self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]
            elif len(integer_var) == 4:
                asset0 = integer_var[0]
                asset1 = integer_var[1]
                asset2 = integer_var[2]
                asset3 = integer_var[3]
                for n, qubit0 in enumerate(self.assets_to_qubits[asset0]):
                    for m, qubit1 in enumerate(self.assets_to_qubits[asset1]):
                        for k, qubit2 in enumerate(self.assets_to_qubits[asset2]):
                            for l, qubit3 in enumerate(self.assets_to_qubits[asset3]):
                                bin_var = ((asset0, qubit0), (asset1, qubit1), (asset2, qubit2), (asset3, qubit3))
                                if self.log_encoding:
                                    self.cost_hubo_bin[bin_var] = 2**(n+m+k+l)*self.cost_hubo_int[integer_var]
                                else:
                                    self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]

    def simplify_cost_hubo_bin(self):
        # Using the fact that x_i^n = x_i, for n > 0, we can simplify the cost function
        self.cost_hubo_bin_simplified = {}
        for bin_var in self.cost_hubo_bin:
            bin_var_set = frozenset(bin_var)
            if bin_var_set in self.cost_hubo_bin_simplified:
                self.cost_hubo_bin_simplified[bin_var_set] += self.cost_hubo_bin[bin_var]
            else:
                self.cost_hubo_bin_simplified[bin_var_set] = self.cost_hubo_bin[bin_var]
        #self.cost_hubo_bin_simplified = self.cost_hubo_bin


    def cost_hubo_bin_to_ising_hamiltonian(self):
        # This function performs the standard x = (1 + z)/2 substitution to convert the binary cost function to an Ising Hamiltonian
        self.hamiltonian = 0
        for bin_var_set in self.cost_hubo_bin_simplified:
            bin_var = list(bin_var_set)
            coeff = self.cost_hubo_bin_simplified[bin_var_set]
            if len(bin_var) == 1:
                qubit0 = bin_var[0][1]
                self.hamiltonian += (coeff/2)*(qml.Identity(qubit0) + qml.PauliZ(qubit0))
            elif len(bin_var) == 2:
                qubit0 = bin_var[0][1]
                qubit1 = bin_var[1][1]
                self.hamiltonian += (coeff/4)*(qml.Identity(qubit0) + qml.PauliZ(qubit0)) @ (qml.Identity(qubit1) + qml.PauliZ(qubit1))
                #self.hamiltonian += (coeff/4)*(qml.Identity(qubit0) @ qml.Identity(qubit1)\
                #                                - qml.Identity(qubit0) @ qml.PauliZ(qubit1)\
                #                                - qml.PauliZ(qubit0) @ qml.Identity(qubit1)\
                #                                + qml.PauliZ(qubit0) @ qml.PauliZ(qubit1))
            elif len(bin_var) == 3:
                qubit0 = bin_var[0][1]
                qubit1 = bin_var[1][1]
                qubit2 = bin_var[2][1]
                #self.hamiltonian += (self.cost_hubo_bin[bin_var]/8)*(qml.Identity(qubit0) - qml.PauliZ(qubit0)) @ (qml.Identity(qubit1) - qml.PauliZ(qubit1)) @ (qml.Identity(qubit2) - qml.PauliZ(qubit2))
                # Multiply the previous line open to speedup pennylane
                self.hamiltonian += (coeff/8)*(qml.Identity(qubit0) @ qml.Identity(qubit1) @ qml.Identity(qubit2)\
                                                + qml.Identity(qubit0) @ qml.Identity(qubit1) @ qml.PauliZ(qubit2)\
                                                + qml.Identity(qubit0) @ qml.PauliZ(qubit1) @ qml.Identity(qubit2)\
                                                + qml.PauliZ(qubit0) @ qml.Identity(qubit1) @ qml.Identity(qubit2)\
                                                + qml.Identity(qubit0) @ qml.PauliZ(qubit1) @ qml.PauliZ(qubit2)\
                                                + qml.PauliZ(qubit0) @ qml.Identity(qubit1) @ qml.PauliZ(qubit2)\
                                                + qml.PauliZ(qubit0) @ qml.PauliZ(qubit1) @ qml.Identity(qubit2)\
                                                + qml.PauliZ(qubit0) @ qml.PauliZ(qubit1) @ qml.PauliZ(qubit2))
                
            elif len(bin_var) == 4:
                qubit0 = bin_var[0][1]
                qubit1 = bin_var[1][1]
                qubit2 = bin_var[2][1]
                qubit3 = bin_var[3][1]
                #self.hamiltonian += (self.cost_hubo_bin[bin_var]/16)*(qml.Identity(qubit0) - qml.PauliZ(qubit0)) @ (qml.Identity(qubit1) - qml.PauliZ(qubit1)) @ (qml.Identity(qubit2) - qml.PauliZ(qubit2)) @ (qml.Identity(qubit3) - qml.PauliZ(qubit3))
                # Multiply the previous line open manually, somehow speedsup the process
                self.hamiltonian += (coeff/16)*(qml.Identity(qubit0) @ qml.Identity(qubit1) @ qml.Identity(qubit2) @ qml.Identity(qubit3)\
                                                + qml.Identity(qubit0) @ qml.Identity(qubit1) @ qml.Identity(qubit2) @ qml.PauliZ(qubit3)\
                                                + qml.Identity(qubit0) @ qml.Identity(qubit1) @ qml.PauliZ(qubit2) @ qml.Identity(qubit3)\
                                                + qml.Identity(qubit0) @ qml.PauliZ(qubit1) @ qml.Identity(qubit2) @ qml.Identity(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.Identity(qubit1) @ qml.Identity(qubit2) @ qml.Identity(qubit3)\
                                                + qml.Identity(qubit0) @ qml.Identity(qubit1) @ qml.PauliZ(qubit2) @ qml.PauliZ(qubit3)\
                                                + qml.Identity(qubit0) @ qml.PauliZ(qubit1) @ qml.Identity(qubit2) @ qml.PauliZ(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.Identity(qubit1) @ qml.Identity(qubit2) @ qml.PauliZ(qubit3)\
                                                + qml.Identity(qubit0) @ qml.PauliZ(qubit1) @ qml.PauliZ(qubit2) @ qml.Identity(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.Identity(qubit1) @ qml.PauliZ(qubit2) @ qml.Identity(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.PauliZ(qubit1) @ qml.Identity(qubit2) @ qml.Identity(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.PauliZ(qubit1) @ qml.PauliZ(qubit2) @ qml.Identity(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.PauliZ(qubit1) @ qml.Identity(qubit2) @ qml.PauliZ(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.Identity(qubit1) @ qml.PauliZ(qubit2) @ qml.PauliZ(qubit3)\
                                                + qml.Identity(qubit0) @ qml.PauliZ(qubit1) @ qml.PauliZ(qubit2) @ qml.PauliZ(qubit3)\
                                                + qml.PauliZ(qubit0) @ qml.PauliZ(qubit1) @ qml.PauliZ(qubit2) @ qml.PauliZ(qubit3))
                

    def get_QAOA_circuits(self):
        #dev = qml.device('qiskit.aer', wires=self.n_qubits)
        dev = qml.device('lightning.qubit', wires=self.n_qubits)
        
        cost_hamiltonian = self.get_cost_hamiltonian()
        mixer_hamiltonian = qml.qaoa.x_mixer(range(self.n_qubits))

        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_hamiltonian)
            qml.qaoa.mixer_layer(alpha, mixer_hamiltonian)
        
        init_guess = [0, 1, 0, 0, 1, 0]
        @qml.qnode(dev)
        def qaoa_circuit(params):
            for wire in range(self.n_qubits):
                #qml.Hadamard(wires=wire)
                if init_guess[wire] == 1:
                    qml.PauliX(wires=wire)
            qml.layer(qaoa_layer, self.layers, params[0], params[1])
            return qml.expval(cost_hamiltonian)
        
        @qml.qnode(dev)
        def qaoa_probs_circuit(params):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, self.layers, params[0], params[1])
            return qml.probs()
        
        # Compile the QAOA circuit to some specific gate set
        # It seems that Pennylane compiler is too eager to decompose, 
        # since it unnecessarily applies the rule RX = H RZ H
        #allowed_gates = ["CNOT", "RZ", "RX", "Hadamard"]
        #dispatched_transform = qml.transform(replace_h_rz_h_with_rx)
        #qaoa_circuit = compile(qaoa_circuit, basis_set = allowed_gates)
        #qaoa_circuit = compile(qaoa_circuit, pipeline = [dispatched_transform])

        #qaoa_probs_circuit = compile(qaoa_probs_circuit, basis_set = allowed_gates)
        #qaoa_probs_circuit = compile(qaoa_probs_circuit, pipeline = [dispatched_transform])

        return qaoa_circuit, qaoa_probs_circuit
    

    def draw_qaoa_circuit(self):
        fig, ax = qml.draw_mpl(self.qaoa_circuit, expansion_strategy="device", decimals=2)(self.init_params)
        fig.savefig("qaoa_circuit.png")


    def solve_exactly(self):
        cost_matrix = self.get_cost_hamiltonian().matrix(wire_order=range(self.n_qubits))
        self.smallest_eigenvalues, self.smallest_eigenvectors, first_excited_energy, first_excited_state = smallest_eigenpairs(cost_matrix)
        self.smallest_bitstrings = [basis_vector_to_bitstring(v) for v in self.smallest_eigenvectors]
        second_smallest_bitstrings = [basis_vector_to_bitstring(first_excited_state)]
        optimized_portfolio = bitstrings_to_optimized_portfolios(self.smallest_bitstrings, self.assets_to_qubits)
        second_optimized_portfolio = bitstrings_to_optimized_portfolios(second_smallest_bitstrings, self.assets_to_qubits)
        return self.smallest_eigenvalues, self.smallest_eigenvectors, self.smallest_bitstrings, first_excited_energy, first_excited_state, optimized_portfolio, second_optimized_portfolio
    

    def solve_with_qaoa(self):
        opt = qml.AdagradOptimizer(stepsize=0.5)
        #opt = qml.QNGOptimizer()
        params = self.init_params.copy()
        probs = self.qaoa_probs_circuit(params)
        total_steps = 2000
        #attempts = 0
        #while True:
        #steps = 10
        for _ in range(total_steps):
            params = opt.step(self.qaoa_circuit, params)
        #total_steps += steps
        probs = self.qaoa_probs_circuit(params)
        #smallest_eigenvalue_now = self.qaoa_circuit(params)
        #most_probable_state = np.argsort(probs)[-1]
        #most_probable_state = int_to_bitstring(most_probable_state, self.n_qubits)
            #print(f"Most probable state: {most_probable_state} and smallest eigenvalue: {smallest_eigenvalue_now} and {self.smallest_bitstrings}")
        """
        if most_probable_state in self.smallest_bitstrings:
            break
        if total_steps > 1000:
            print("Optimization did not converge")
            print("Trying with a new initialization")
            self.init_params = np.pi*np.random.rand(2, self.layers, requires_grad=True)
            params = self.init_params.copy()
            total_steps = 0
            attempts += 1
        if attempts > 10:
            print("Optimization did not converge to the known optimal solution after ", attempts, " attempts.")
            print("Returning the best solution found so far")
            break
        """

        #probs = self.qaoa_probs_circuit(params)
        final_expectation_value = self.qaoa_circuit(params)
        two_most_probable_states = np.argsort(probs)[-2:]
        states_probs = [probs[i] for i in two_most_probable_states]
        two_most_probable_states = [int_to_bitstring(i, self.n_qubits) for i in two_most_probable_states]
        optimized_portfolios = bitstrings_to_optimized_portfolios(two_most_probable_states, self.assets_to_qubits)
        return two_most_probable_states, final_expectation_value, params, total_steps, states_probs, optimized_portfolios


    def get_assets_to_qubits(self):
        return self.assets_to_qubits
    
    def get_qubits_to_assets(self):
        return self.qubits_to_assets
    
    def get_cost_hubo_int(self):
        return self.cost_hubo_int
    
    def get_cost_hubo_bin(self):
        return self.cost_hubo_bin
    
    def get_cost_hubo_bin_simplified(self):
        return self.cost_hubo_bin_simplified
    
    def get_cost_hamiltonian(self):
        return self.hamiltonian
    

    def solve_with_iterative_QAOA(self, max_layers = 10):
        # Start with layers = 1 and increase the number of layers until convergence and use the previous layers optimized params as initialization
        for layers in range(1, max_layers + 1):
            print(f"Trying with {layers} layers")
            self.layers = layers
            self.qaoa_circuit, self.qaoa_probs_circuit = self.get_QAOA_circuits()
            opt = qml.AdagradOptimizer(stepsize=(1/(layers+1)))
            if layers == 1:
                params = 0.01*np.random.rand(2, self.layers, requires_grad=True)
            else:
                params = np.array([[p for p in params[0]] + [params[0][-1]], [p for p in params[1]] + [params[1][-1]]], requires_grad=True)
            
            total_steps = 500*layers
            for _ in range(total_steps):
                params = opt.step(self.qaoa_circuit, params)
            
            # Print the results
            probs = self.qaoa_probs_circuit(params)
            final_expectation_value = self.qaoa_circuit(params)
            two_most_probable_states = np.argsort(probs)[-2:]
            states_probs = [probs[i] for i in two_most_probable_states]
            two_most_probable_states = [int_to_bitstring(i, self.n_qubits) for i in two_most_probable_states]
            optimized_portfolios = bitstrings_to_optimized_portfolios(two_most_probable_states, self.assets_to_qubits)
            print(f"Two most probable states: {two_most_probable_states} with probabilities {states_probs}")
            print(f"Final expectation value: {final_expectation_value}")
            print(f"Optimized portfolios: {optimized_portfolios}")
            if layers > 1:
                if np.isclose(final_expectation_value, previous_expectation_value, rtol=1e-03):
                    break
            previous_expectation_value = final_expectation_value

        return two_most_probable_states, final_expectation_value, params, total_steps, states_probs, optimized_portfolios

