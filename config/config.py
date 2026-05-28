import os

# Environment: local, dev, qa, stage, prod - affects behavior at runtime (for future enhancements)
ENV = os.environ.get("ENV", default="local")

# Host/Port the Flask app binds to.
API_HOST = "0.0.0.0"
API_PORT = 5000

# Number of decimal places for time metadata in responses.
TIME_DECIMALS = 6
