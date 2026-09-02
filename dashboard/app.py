#!/usr/bin/env python3
"""
Verit NIDS - Dashboard Backend
------------------------------------------------------------------
Serves the live monitoring dashboard: the static single-page UI, plus
three small endpoints it talks to:

    GET /api/stats   -- summary counters (badges)
    GET /api/recent   -- last N flow results, for the table's initial load
    GET /api/stream   -- Server-Sent Events: pushes each new flow result
                          as it happens

Runs in a background thread inside the same process as the detection
engine (see hybrid_detect.py) -- no separate service, no network hop,
true real-time push with no polling latency.
"""

import json
import queue
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(event_bus):
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(_STATIC_DIR, "index.html")

    @app.route("/api/stats")
    def api_stats():
        return jsonify(event_bus.get_stats())

    @app.route("/api/recent")
    def api_recent():
        return jsonify(event_bus.get_recent(150))

    @app.route("/api/stream")
    def api_stream():
        client_queue = event_bus.subscribe()

        def event_stream():
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        event = client_queue.get(timeout=15)
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        yield ": heartbeat\n\n"  # keeps the connection alive through proxies/timeouts
            except GeneratorExit:
                pass
            finally:
                event_bus.unsubscribe(client_queue)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run_dashboard(event_bus, host="0.0.0.0", port=8080):
    app = create_app(event_bus)
    print(f"[dashboard] Serving on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
