import logging
import sys

# Get the root logger
root_logger = logging.getLogger()

# Check if handlers already exist to avoid duplicates
if not root_logger.handlers:
    # Create a handler that writes to sys.stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(handler)

# Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
root_logger.setLevel(logging.INFO)