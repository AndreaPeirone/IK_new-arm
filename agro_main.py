import numpy as np
import matplotlib.pyplot as plt

def main():
    i =0
    while i < 10:
        # Create a simple plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)

        plt.plot(x, y, label=f'Sine Wave {i}')
        plt.title('Sine Waves')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        i += 1 