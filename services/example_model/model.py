import numpy as np
import threading
import time

from .config import N_TH_ROOT
from .schema import InputSchema
from .version import CODE_VERSION, WEIGHTS_VERSION


class ExampleModel:
    """Example ML model: calculates the N-th root of the input value.
    N_TH_ROOT is loaded from config.py (default: 2 = square root).
    Uses a threading lock to protect the computation path."""

    def __init__(self):
        self.mutex = threading.Lock()
        self.ready_state = True

    def predict(self, payload):
        """Run the model's prediction on the input payload."""
        # Preprocess the input
        inputs = payload.get('input')
        n_th_root = payload.get('root')  #, N_TH_ROOT)  # default nth root is provided by the schema

        with self.mutex:
            time.sleep(0.2 * np.random.random())  # Simulate some processing time

            if isinstance(inputs, list):
                results = [(float(v) ** (1 / float(n_th_root))) for v in inputs]
            else:
                results = [float(inputs) ** (1 / float(n_th_root))]
            #result = float(value) ** (1 / float(n_th_root))
        
        # Post-process
        result = [float(v) for v in results]
        
        return {
            'result': result,
            'metadata': {
                'service': 'example_model',
                'request_id': payload.get('request_id', None)
            }
        }

    def health_check(self):
        """Return the service's health status."""
        return self.ready_state

    @staticmethod
    def get_service_type():
        """Return the service type identifier for this model."""
        return "model"

    @staticmethod
    def get_version():
        """Return the combined code version and weights version."""
        return '-'.join([CODE_VERSION, WEIGHTS_VERSION])

    @staticmethod
    def get_input_schema():
        """Return the Pydantic input schema class for this model."""
        return InputSchema
