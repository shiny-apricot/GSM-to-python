"""
File's primary purpose and role in the pipeline:
This module provides utility functions for visualizing data distributions and relationships in the context of bioinformatics analysis.

Key functions:
- plot_histogram: Plots a histogram of the given data.
- plot_scatter: Creates a scatter plot for two variables.
- plot_box: Generates a box plot for visualizing data distributions.

Usage examples:
    plot_histogram(data, bins=30)
    plot_scatter(x, y)
    plot_box(data)

Important notes:
- Ensure that matplotlib is installed in your environment.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """Plots a histogram of the given data."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_scatter(x, y, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
    """Creates a scatter plot for two variables."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='green', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_box(data, title='Box Plot', xlabel='Categories', ylabel='Values'):
    """Generates a box plot for visualizing data distributions."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()
