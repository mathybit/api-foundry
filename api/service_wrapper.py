import json
import time
from flask import request
from pydantic import ValidationError

from config.config import TIME_DECIMALS
from version import API_VERSION


class APIServiceWrapper:
    """
    Wraps a service class (compute service or ML model) with API-level concerns:
    input validation, timing, versioning, and type checking.
    """

    def __init__(self, service_class):
        self.service = service_class()

    def compute(self, payload):
        """
        Run the service's compute() method. Returns JSON string.
        Adds total_time, api_version, and service_version to the response metadata.
        Validates input via Pydantic schema if the service provides one.
        """
        if not hasattr(self.service, 'compute'):
            return json.dumps({"error": f"The compute endpoint is a service endpoint that is not available for {self.service.__class__.__name__}"})

        # Validate payload against the service's input schema
        input_schema = self.get_input_schema(as_json=False)
        if input_schema:
            payload = input_schema.model_validate(payload).model_dump(mode="json")

        t0 = time.time()
        response = self.service.compute(payload)
        t1 = time.time()

        if "metadata" not in response:
            response["metadata"] = {}
        response["metadata"].update({
            "total_time": round(t1 - t0, TIME_DECIMALS),
            "api_version": API_VERSION,
            "service_version": self.service.get_version(),
        })

        return json.dumps(response)

    def predict(self, payload):
        """
        Run the service's predict() method. Returns JSON string.
        Adds total_time, api_version, and service_version to the response metadata.
        Validates input via Pydantic schema if the service provides one.
        """
        if not hasattr(self.service, 'predict'):
            return json.dumps({"error": f"The predict endpoint is a model endpoint that is not available for {self.service.__class__.__name__}"})

        # Validate payload against the service's input schema
        input_schema = self.get_input_schema(as_json=False)
        if input_schema:
            payload = input_schema.model_validate(payload).model_dump(mode="json")

        t0 = time.time()
        response = self.service.predict(payload)
        t1 = time.time()

        if "metadata" not in response:
            response["metadata"] = {}
        response["metadata"].update({
            "total_time": round(t1 - t0, TIME_DECIMALS),
            "api_version": API_VERSION,
            "service_version": self.service.get_version(),
        })

        return json.dumps(response)

    def get_input_schema(self, as_json=True):
        """
        Return the service's input schema.
        If `as_json=True`, the retruned schema is a JSON dict (used in /schema endpoint).
        If `as_json=False`, the returned schema is a Pydantic model (used in validation).
        """
        schema = self.service.get_input_schema()
        if as_json:
            if hasattr(schema, "model_json_schema"):  # Handle Pydantic models - get the JSON Schema dict from the class
                schema = schema.model_json_schema()
            return schema
        else:
            if hasattr(schema, "model_validate"):
                return schema
        return None

    def get_service_type(self):
        """Delegate to the service's get_service_type() - returns 'service' or 'model'."""
        return self.service.get_service_type()

    def get_version(self):
        """Combine API version and service version, e.g. '2026-05-21-01.2026-05-23-01'."""
        return '.'.join([API_VERSION, self.service.get_version()])

    def health_check(self):
        """Delegate to the service's health_check() - should return True or False."""
        return self.service.health_check()
