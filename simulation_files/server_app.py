"""my-app: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from my_app.strategy import SaveModelStrategy

from pathlib import Path
from my_app.task import load_model

import numpy as np
# Make TensorFlow log less verbose
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # THIS CAN BE INITIALIZED WITH PRETRAINED MODEL TO CONTINUE TRAINING FROM CHECKPOINT
    # Note: The data will be shuffled again when starting from checkpoint
    #npz_path = Path.cwd() / "model-weights10.npz"
    #npz_data = np.load(npz_path)
    
    # Convert NPZ file contents to list of arrays
    # NPZ files store arrays in a dict-like format
    #weights = [npz_data[key] for key in npz_data.files]
    #parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=4,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
