from typing import Optional, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes, Scalar

import flwr as fl
import numpy as np

import json

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """ FedAvg strategy with model and loss saving feature added 
        
        Other stategies can be copied from Flwr and same functions 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialises JSON file
        self.json_path = "results.json"
        with open(self.json_path, 'w') as f:
            json.dump({"server_losses": []}, f)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"model-weights{server_round}.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Handles saving losses to JSON file
        results_dict = {"epoch": server_round,
                       "loss": loss}
        
        print("SAVING LOSSES TO JSON")
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            data["server_losses"].append(results_dict)

            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving server losses to JSON: {e}")
            
        return loss, metrics
