import http.server
import socketserver
import webbrowser
import os
import threading

# --- Configuration ---
PORT = 8000
HOSTNAME = "localhost"
# The script will now look for this specific HTML file.
HTML_FILENAME = "app_interface.html" 
MODEL_FILES = [
    "anxiety_stacking_model_tuned.onnx",
    "anxiety_stacking_model.onnx"
]

def check_and_prepare_files():
    """
    Checks for the required HTML file and creates dummy model files
    if the real ones don't exist.
    """
    # 1. Check if the main HTML file exists.
    if not os.path.exists(HTML_FILENAME):
        print("="*60)
        print(f"!!! ERROR: File not found: '{HTML_FILENAME}'")
        print("Please make sure your HTML file is saved with that exact name")
        print("in the same directory as this Python script.")
        print("="*60)
        return False # Indicate failure
    else:
        print(f"✅ Found interface file: '{HTML_FILENAME}'")

    # 2. Check for ONNX model files and create placeholders if missing.
    for model_file in MODEL_FILES:
        if not os.path.exists(model_file):
            print(f"⚠️  Creating dummy model file: '{model_file}'...")
            print(f"   [NOTE] This is a placeholder. The app's analysis will likely fail.")
            print(f"   [ACTION] For the app to work, place the real '{model_file}' here.")
            with open(model_file, "w") as f:
                f.write("This is a dummy file.")
        else:
            print(f"✅ Found model file: '{model_file}'")
    
    return True # Indicate success

def start_server():
    """Starts a local HTTP server and opens the web browser to the correct page."""
    
    # Use a custom handler to keep the console output clean.
    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress the normal 'GET /... 200 OK' logs for a cleaner terminal
            if "200" not in args and "304" not in args:
                 super().log_message(format, *args)

    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer((HOSTNAME, PORT), QuietHTTPRequestHandler) as httpd:
        # Construct the full URL to your specific HTML file
        server_url = f"http://{HOSTNAME}:{PORT}/{HTML_FILENAME}"
        
        print("\n" + "="*50)
        print(f"Server starting up...")
        print(f"Serving files from: {os.getcwd()}")
        print(f"Opening your browser to: {server_url}")
        print("\nTo stop the server, press Ctrl+C in this terminal.")
        print("="*50 + "\n")
        
        # Open the correct URL in a new browser tab
        threading.Timer(1, lambda: webbrowser.open_new_tab(server_url)).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down the server...")
            httpd.shutdown()

if __name__ == "__main__":
    # First, check if all required files are present.
    if check_and_prepare_files():
        # If they are, start the server.
        start_server()