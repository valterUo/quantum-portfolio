
"""An application class for a portfolio optimization problem."""
from typing import List, Tuple, Union, Optional

import numpy as np

#from docplex.mp.advmodel import AdvModel
#from qiskit_optimization.algorithms import OptimizationResult
#from qiskit_optimization.problems import QuadraticProgram
#from qiskit_optimization.translators import from_docplex_mp
#from qiskit_finance.exceptions import QiskitFinanceError


class PortfolioOptimization():

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariances: np.ndarray,
        skewness: np.ndarray,
        kurtosis: np.ndarray,
        risk_factor: float,
        budget: int,
        bounds: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """
        Args:
            expected_returns: The expected returns for the assets.
            covariances: The covariances between the assets.
            risk_factor: The risk appetite of the decision maker.
            budget: The budget, i.e. the number of assets to be selected.
            bounds: The list of tuples for the lower bounds and the upper bounds of each variable.
                e.g. [(lower bound1, upper bound1), (lower bound2, upper bound2), ...].
                Default is None which means all the variables are binary variables.
        """
        self._expected_returns = expected_returns
        self._covariances = covariances
        self._risk_factor = risk_factor
        self._budget = budget
        self._bounds = bounds
        self._check_compatibility(bounds)

    def to_hubo_ocean(self):
        pass
    
    def to_Hamiltonian_pennylane(self):
        pass
    
    def to_Pauli_decomposition_pennylane(self):
        pass
        

    def portfolio_expected_value(self, result: Union[OptimizationResult, np.ndarray]) -> float:
        """Returns the portfolio expected value based on the result.

        Args:
            result: The calculated result of the problem

        Returns:
            The portfolio expected value
        """
        x = self._result_to_x(result)
        return np.dot(self._expected_returns, x)


    def portfolio_variance(self, result: Union[OptimizationResult, np.ndarray]) -> float:
        """Returns the portfolio variance based on the result

        Args:
            result: The calculated result of the problem

        Returns:
            The portfolio variance
        """
        x = self._result_to_x(result)
        return np.dot(x, np.dot(self._covariances, x))


    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as a list of asset indices

        Args:
            result: The calculated result of the problem

        Returns:
            The list of asset indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, x_i in enumerate(x) if x_i]


    def _check_compatibility(self, bounds) -> None:
        """Check the compatibility of given variables"""
        if len(self._expected_returns) != len(self._covariances) or not all(
            len(self._expected_returns) == len(row) for row in self._covariances
        ):
            raise QiskitFinanceError(
                "The sizes of expected_returns and covariances do not match. ",
                f"expected_returns: {self._expected_returns}, covariances: {self._covariances}.",
            )
        if bounds is not None:
            if (
                not isinstance(bounds, list)
                or not all(isinstance(lb_, int) for lb_, _ in bounds)
                or not all(isinstance(ub_, int) for _, ub_ in bounds)
            ):
                raise QiskitFinanceError(
                    f"The bounds must be a list of tuples of integers. {bounds}",
                )
            if any(ub_ < lb_ for lb_, ub_ in bounds):
                raise QiskitFinanceError(
                    "The upper bound of each variable, in the list of bounds, must be greater ",
                    f"than or equal to the lower bound. {bounds}",
                )
            if len(bounds) != len(self._expected_returns):
                raise QiskitFinanceError(
                    f"The lengths of the bounds, {len(self._bounds)}, do not match to ",
                    f"the number of types of assets, {len(self._expected_returns)}.",
                )

    @property
    def expected_returns(self) -> np.ndarray:
        """Getter of expected_returns

        Returns:
            The expected returns for the assets.
        """
        return self._expected_returns

    @expected_returns.setter
    def expected_returns(self, expected_returns: np.ndarray) -> None:
        """Setter of expected_returns

        Args:
            expected_returns: The expected returns for the assets.
        """
        self._expected_returns = expected_returns

    @property
    def covariances(self) -> np.ndarray:
        """Getter of covariances

        Returns:
            The covariances between the assets.
        """
        return self._covariances

    @covariances.setter
    def covariances(self, covariances: np.ndarray) -> None:
        """Setter of covariances

        Args:
            covariances: The covariances between the assets.
        """
        self._covariances = covariances

    @property
    def risk_factor(self) -> float:
        """Getter of risk_factor

        Returns:
            The risk appetite of the decision maker.
        """
        return self._risk_factor

    @risk_factor.setter
    def risk_factor(self, risk_factor: float) -> None:
        """Setter of risk_factor

        Args:
            risk_factor: The risk appetite of the decision maker.
        """
        self._risk_factor = risk_factor

    @property
    def budget(self) -> int:
        """Getter of budget

        Returns:
            The budget, i.e. the number of assets to be selected.
        """
        return self._budget

    @budget.setter
    def budget(self, budget: int) -> None:
        """Setter of budget

        Args:
            budget: The budget, i.e. the number of assets to be selected.
        """
        self._budget = budget

    @property
    def bounds(self) -> List[Tuple[int, int]]:
        """Getter of the lower bounds and upper bounds of each selectable assets.

        Returns:
            The lower bounds and upper bounds of each assets selectable
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: List[Tuple[int, int]]) -> None:
        """Setter of the lower bounds and upper bounds of each selectable assets.

        Args:
            bounds: The lower bounds and upper bounds of each assets selectable
        """
        self._check_compatibility(bounds)  # check compatibility before setting bounds
        self._bounds = bounds