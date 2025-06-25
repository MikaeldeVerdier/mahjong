import csv
from ultralytics.utils.plotting import plot_results

class ModelPlotter:
    def __init__(self, results_path="results.csv"):
        self.results_path = results_path

    def create_results(self, model):  # create from checkpoint if not already created
            results = model.ckpt["train_results"]
            with open(self.results_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(results.keys())
                for i in range(len(results["epoch"])):  # write each row
                    row = [results[key][i] for key in results.keys()]
                    writer.writerow(row)

    def plot_results(self):
        plot_results(self.results_path)
