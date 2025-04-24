import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load .env file before potentially importing parts of the app
load_dotenv()

# Ensure the backend directory is in the path if needed for discovery
# (uvicorn handles module path well usually, but can be explicit)
# backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
# if backend_dir not in sys.path:
#     sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    # Determine the number of workers based on CPU count.
    # A common recommendation is (2 * number_of_cores) + 1.
    # Uvicorn/Gunicorn manage workers correctly even with reload=True.
    cpu_cores = os.cpu_count() or 1  # Default to 1 if cpu_count() returns None
    print(f"Detected {cpu_cores} CPU cores.")
    workers = (2 * cpu_cores) + 1
    print(f"Setting workers = {workers}")
    use_reload = False  # Enable auto-reload for development

    # Get host and port from environment variables or use defaults
    host = os.getenv("BACKEND_HOST", "127.0.0.1")  # Default to localhost
    port = int(os.getenv("BACKEND_PORT", "8888"))  # Default to 8888

    print(f"Starting backend server on {host}:{port}...")
    print(f"Reloading enabled: {use_reload}")
    print(f"Detected {cpu_cores} CPU cores. Setting workers = {workers}")
    # Run the Uvicorn server
    # The application is specified as 'module:app_instance'
    # Example: "backend.main:app"
    uvicorn.run(
        "backend.socket:app",
        host=host,
        port=port,
        reload=use_reload,
        workers=workers
        # log_level="info" # Can adjust log level
    )
