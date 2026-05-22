import numpy as np
import time

from .config import N_TH_ROOT
from .schema import InputSchema
from .version import CODE_VERSION


class ExampleService:
    def __init__(self):
        self.ready_state = True

    def compute(self, payload):
        # Preprocess the input
        value = payload.get('input', 0)

        time.sleep(0.2 * np.random.random())  # Simulate some processing time
        result = float(value) ** (1 / N_TH_ROOT)
        
        return {
            'result': float(result),
            'metadata': {
                'service': 'example_service',
                'request_id': payload.get('request_id', None)
            }
        }

    def health_check(self):
        return self.ready_state
    
    @staticmethod
    def get_service_type():
        return "service"

    @staticmethod
    def get_version():
        return CODE_VERSION

    @staticmethod
    def get_input_schema():
        return InputSchema
