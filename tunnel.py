"""
D-SAAT Cloudflare HTTPS Tunnel Runner

Spawns a zero-config Cloudflare edge tunnel to localhost:8000,
extracts the generated public HTTPS URL, and launches the browser.
Allows anyone on mobile, tablet, or remote PC to test D-SAAT live online.
"""

import os
import re
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parent
TOOLS_DIR = ROOT / "tools"
TOOLS_DIR.mkdir(exist_ok=True)

CF_EXE = TOOLS_DIR / "cloudflared.exe"
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"


def ensure_cloudflared():
    if not CF_EXE.exists():
        print("[INFO] cloudflared.exe not found. Downloading Cloudflare Tunnel binary...", flush=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(CF_URL, str(CF_EXE))
            print(f"[OK] Downloaded to {CF_EXE}", flush=True)
        except Exception as exc:
            print(f"[ERROR] Failed to download cloudflared: {exc}", flush=True)
            sys.exit(1)


def ensure_server():
    """Ensure backend server is running on port 8000."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = sock.connect_ex(('127.0.0.1', 8000)) == 0
    sock.close()
    if not is_open:
        print("[INFO] Starting D-SAAT server in background on port 8000...", flush=True)
        subprocess.Popen([sys.executable, "web_app.py", "--no-browser"], cwd=str(ROOT))
        for _ in range(15):
            time.sleep(1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if sock.connect_ex(('127.0.0.1', 8000)) == 0:
                sock.close()
                print("[OK] Backend server is active.", flush=True)
                return
            sock.close()
        print("[WARN] Server did not report ready within 15s; proceeding with tunnel anyway.", flush=True)
    else:
        print("[OK] Backend server is already running on port 8000.", flush=True)


def start_tunnel():
    ensure_cloudflared()
    ensure_server()

    print("\n" + "=" * 65, flush=True)
    print("  CONNECTING TO CLOUDFLARE EDGE NETWORK (HTTPS TUNNEL)", flush=True)
    print("=" * 65, flush=True)
    print("[INFO] Requesting secure public HTTPS URL from Cloudflare...", flush=True)

    proc = subprocess.Popen(
        [str(CF_EXE), "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    public_url = None

    try:
        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.1)
                continue

            match = url_regex.search(line)
            if match:
                public_url = match.group(0)
                print("\n" + "=" * 65, flush=True)
                print("  [SUCCESS] D-SAAT IS LIVE ONLINE WITH FULL HTTPS ACCESS:", flush=True)
                print(f"     PUBLIC URL :  {public_url}", flush=True)
                print("     LOCAL URL  :  http://localhost:8000", flush=True)
                print("=" * 65, flush=True)
                print("  [*] Share this link with any smartphone, laptop, or tablet.", flush=True)
                print("  [*] Click 'Browser Cam: OFF' in the top bar to stream their camera.", flush=True)
                print("  [*] Press Ctrl+C at any time to disconnect the tunnel.\n", flush=True)
                try:
                    webbrowser.open(public_url)
                except Exception:
                    pass
                break

        # Keep running until user stops
        proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Cloudflare tunnel...", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("[OK] Cloudflare tunnel stopped cleanly. Goodbye!", flush=True)


if __name__ == "__main__":
    start_tunnel()
