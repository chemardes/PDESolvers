from abc import ABC, abstractmethod

class Solver(ABC):

    @abstractmethod
    def solve(self):
        """
        Solves the equation with the chosen solver and returns the solution
        """
        pass
