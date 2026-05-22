import json
import time
from flask import request
from pydantic import ValidationError

from config.config import TIME_DECIMALS
from version import API_VERSION


class APIServiceWrapper:
    def __init__(self, service_class):
        self.service = service_class()

    def compute(self, payload):
        if not hasattr(self.service, 'compute'):
            return json.dumps({"error": f"The compute endpoint is a service endpoint that is not available for {self.service.__class__.__name__}"})

        # Validate payload against the service's input schema
        input_model = self.get_input_model()
        if input_model:
            payload = input_model.model_validate(payload).model_dump(mode="json")

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
        if not hasattr(self.service, 'predict'):
            return json.dumps({"error": f"The predict endpoint is a model endpoint that is not available for {self.service.__class__.__name__}"})

        # Validate payload against the service's input schema
        input_model = self.get_input_model()
        if input_model:
            payload = input_model.model_validate(payload).model_dump(mode="json")

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
        schema = self.service.get_input_schema()
        # Handle Pydantic models — get the JSON Schema dict from the class
        if hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        if as_json:
            return json.dumps(schema)
        return schema

    def get_input_model(self):
        schema = self.service.get_input_schema()
        if hasattr(schema, "model_validate"):
            return schema
        return None

    def get_service_type(self):
        return self.service.get_service_type()

    def get_version(self):
        return '.'.join([API_VERSION, self.service.get_version()])

    def health_check(self):
        return self.service.health_check()
