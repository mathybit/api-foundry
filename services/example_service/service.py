import numpy as np
import time

from .config import N_TH_ROOT
from .schema import InputSchema
from .version import CODE_VERSION


class ExampleService:
    """Example compute service: calculates the N-th root of the input value.
    N_TH_ROOT is loaded from config.py (default: 2 = square root)."""

    def __init__(self):
        self.ready_state = True

    def compute(self, payload):
        """Compute the N-th root of the input value."""
        # Preprocess the input
        value = payload.get('value', 0)
        n_th_root = payload.get('root')  #, N_TH_ROOT)  # default nth root is provided by the schema

        time.sleep(0.2 * np.random.random())  # Simulate some processing time
        result = float(value) ** (1 / float(n_th_root))
        
        return {
            'result': float(result),
            'metadata': {
                'service': 'example_service',
                'request_id': payload.get('request_id', None)
            }
        }

    def health_check(self):
        """Return the service's health status."""
        return self.ready_state

    @staticmethod
    def get_service_type():
        """Return the service type identifier for this service."""
        return "service"

    @staticmethod
    def get_version():
        """Return the service code version string."""
        return CODE_VERSION

    @staticmethod
    def get_input_schema():
        """Return the Pydantic input schema class for this service."""
        return InputSchema
