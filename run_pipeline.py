from pipelines.training_pipeline import training_pipelines
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipelines(data_path='data/processed_olist_customer_dataset.csv')