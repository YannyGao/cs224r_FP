import matplotlib.pyplot as plt


def plot_graph(title: str, x_label: str, x: list[int], y_label: str, y: list[int]):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_hist(data: list[int], ticks: int):
    plt.hist(data, rwidth=1.0)
    plt.xticks(range(ticks))
    plt.show()
