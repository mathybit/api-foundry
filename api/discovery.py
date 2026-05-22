import importlib.util
import json
import os
from pathlib import Path

from flask import Blueprint, g, jsonify, request
from pydantic import ValidationError
from .service_wrapper import APIServiceWrapper


_service_registry = {}


def create_service_blueprints():
    bp = Blueprint("service_routes", __name__, url_prefix="/service/<service_name>")

    @bp.url_value_preprocessor
    def _resolve_service(endpoint, values):
        #print('_resolve_service() called:', endpoint, values)
        g.service_name = values.get("service_name")
        g.wrapper_instance = _service_registry.get(g.service_name)

    @bp.route("/predict", methods=["POST"])
    def predict(service_name=None):
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
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        try:
            result = g.wrapper_instance.get_input_schema()
            return jsonify(json.loads(result))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route("/version", methods=["GET"])
    def version(service_name=None):
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        return jsonify({"version": g.wrapper_instance.get_version()})

    @bp.route("/health_check", methods=["GET"])
    def health_check(service_name=None):
        if not g.get("wrapper_instance"):
            return jsonify({"error": "Service not found"}), 404
        return jsonify({"healthy": g.wrapper_instance.health_check()})

    return bp


def discover_services(services_dir, app):
    """Discover services, create wrappers, register them, and return registry info."""
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
        _service_registry[entry.name] = wrapper  # register the service
        registry[entry.name] = {
            "name": entry.name,
            "type": wrapper.get_service_type(),
            "version": wrapper.get_version()
        }

    return registry
