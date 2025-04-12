import numpy as np
import matplotlib.pyplot as plt

class BrownianMotion:

    def __init__(self, T, time_steps, sim, mu=0, sigma=1):
        self.__mu = mu
        self.__sigma = sigma
        self.__T = T
        self.__time_steps = time_steps
        self.__sim = sim
        self.__B = None

    def simulate_bm(self):
        t = self.__generate_grid()
        dt = t[1] - t[0]

        Z = np.random.normal(0,1, (self.__sim, self.__time_steps))
        B = np.zeros((self.__sim, self.__time_steps))

        for i in range(self.__sim):
            for j in range(1, self.__time_steps):
                B[i,j] = B[i,j-1] + self.__mu * dt + self.__sigma * np.sqrt(dt) * Z[i,j]

        self.__B = B
        return self


    def __generate_grid(self):
        """
        Generate a time grid from 0 to T with `time_steps` intervals.

        Returns:
        - A numpy array representing the time grid.
        """

        return np.linspace(0, self.__T, self.__time_steps)

    def __get_paths(self):
        return self.__B

    def plot(self):
        t = self.__generate_grid()
        B = self.__get_paths()

        plt.figure(figsize=(10,6))
        for i in range(self.__sim):
            plt.plot(t, B[i], color='grey', alpha=0.3)

        plt.title("Simulated Brownian Motion Paths")
        plt.xlabel("Time (Years)")
        plt.ylabel("B(t)")
        # plt.show()
        plt.savefig("brownian_plot.pdf", bbox_inches='tight')


def main():
    BrownianMotion(T=1, time_steps=365, sim=200).simulate_bm().plot()

if __name__ == "__main__":
    main()