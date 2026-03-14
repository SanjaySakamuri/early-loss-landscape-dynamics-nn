from train import train_model
from failure_predictor import FailurePredictor

def run_experiment():

    predictor = FailurePredictor()

    for i in range(10):
        model, features = train_model()

        label = "stable"

        predictor.add_example(features, label)

    predictor.train()
    print("Predictor trained")

if __name__ == "__main__":
    run_experiment()

