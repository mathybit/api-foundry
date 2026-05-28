import importlib.util
import json
import os
from pathlib import Path

from flask import Blueprint, g, jsonify, request
from pydantic import ValidationError
from .service_wrapper import APIServiceWrapper


# Global registry mapping service name string to APIServiceWrapper instance.
# Shared with the blueprint so routes can look up the instance by service_name.
_service_registry = {}


def create_service_blueprints():
    """
    Create a single shared Flask blueprint for all service routes.
    URL prefix: /service/<service_name>.
    Registers the following prefixes:
    * /service/<service_name>/compute
    * /service/<service_name>/predict
    * /service/<service_name>/schema
    * /service/<service_name>/version
    * /service/<service_name>/health_check
    """
    bp = Blueprint("service_routes", __name__, url_prefix="/service/<service_name>")

    @bp.url_value_preprocessor
    def _resolve_service(endpoint, values):
        """Populate Flask's g object with the service name and wrapper instance
        extracted from the URL path."""
        #print('_resolve_service() called:', endpoint, values)
        g.service_name = values.get("service_name")
        g.wrapper_instance = _service_registry.get(g.service_name)

    @bp.route("/predict", methods=["POST"])
    def predict(service_name=None):
        """Route for ML model predictions. Validates input via Pydantic schema
        (if provided by the service) and delegates to the wrapper's predict()."""
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        try:
            payload = request.get_json(force=False, silent=True) or {}
            result = g.wrapper_instance.predict(payload)
            return jsonify(json.loads(result))
        except ValidationError as e:
            return jsonify({"error": "Validation failed", "details": e.errors()}), 422
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route("/compute", methods=["POST"])
    def compute(service_name=None):
        """Route for general compute services. Validates input via Pydantic schema
        (if provided by the service) and delegates to the wrapper's compute()."""
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        try:
            payload = request.get_json(force=False, silent=True) or {}
            result = g.wrapper_instance.compute(payload)
            return jsonify(json.loads(result))
        except ValidationError as e:
            return jsonify({"error": "Validation failed", "details": e.errors()}), 422
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route("/schema", methods=["GET"])
    def schema(service_name=None):
        """Return the service's JSON input schema."""
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        try:
            schema = g.wrapper_instance.get_input_schema(as_json=True)
            return jsonify(schema)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route("/version", methods=["GET"])
    def version(service_name=None):
        """Return the combined API + service version string."""
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        return jsonify({"version": g.wrapper_instance.get_version()})

    @bp.route("/health_check", methods=["GET"])
    def health_check(service_name=None):
        """Return the service's health status (True = healthy)."""
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        return jsonify({"healthy": g.wrapper_instance.health_check()})

    return bp


def discover_services(services_dir, app):
    """
    Scan the `services` directory, load each subpackage, create APIServiceWrapper
    instances, register them in _service_registry, and return a dict of service metadata.
    """
    registry = {}
    base_bp = create_service_blueprints()
    app.register_blueprint(base_bp)

    for entry in sorted(os.scandir(str(services_dir)), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        init_path = Path(entry.path) / "__init__.py"
        if not init_path.exists():
            print(f"Warning: No __init__.py found in {entry.path}")
            continue

        spec = importlib.util.spec_from_file_location(
            f"services.{entry.name}", str(init_path))
        pkg = importlib.util.module_from_spec(spec)
        pkg.__package__ = f"services.{entry.name}"
        spec.loader.exec_module(pkg)

        cls = getattr(pkg, "ServiceClass", None)
        if cls is None:
            continue

        wrapper = APIServiceWrapper(cls)
        _service_registry[entry.name] = wrapper
        registry[entry.name] = {
            "name": entry.name,
            "type": wrapper.get_service_type(),
            "version": wrapper.get_version()
        }

    return registry
