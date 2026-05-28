from flask import Flask, jsonify
import importlib.util
import os
from pathlib import Path

from api.discovery import discover_services
import config.config as main_config
from version import API_VERSION


app = Flask(__name__)
app.config.from_object(main_config)

# --- Service discovery ---
services_dir = Path(__file__).parent / "services"
service_registry = discover_services(services_dir, app)


@app.route("/version")
def version():
    """Return the API-level CODE_VERSION."""
    return jsonify({"version": API_VERSION})


@app.route("/services")
def list_services():
    """Return a JSON list of all discovered services with their names, types, and versions."""
    return jsonify(list(service_registry.values()))


# --- Start ---
if __name__ == "__main__":
    print(f"Starting API on port {app.config.get('API_PORT', 5000)}")
    print(f"Discovered {len(service_registry)} service(s):")
    for name, info in service_registry.items():
        print(f"  - {name} ({info['type']})")

    #print(f"Trusted hosts: {app.config['TRUSTED_HOSTS']}")
    #print(f"Server name: {app.config.get('SERVER_NAME')}")
    for rule in app.url_map.iter_rules():
        print(f"Rule: {rule} | Endpoint: {rule.endpoint}")
    app.run(host=app.config.get("API_HOST", "0.0.0.0"), port=app.config.get("API_PORT", 5000))
