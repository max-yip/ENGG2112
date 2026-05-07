from lib.training_tools import (
    ExperimentTracker, plot_results, Experiment
)


def main():
    tracker = ExperimentTracker()
    plot_results(tracker)

if __name__ == '__main__':
    main()