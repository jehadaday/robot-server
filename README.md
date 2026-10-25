# robot-server
ESP32 voice robot server (Gemini + edge-tts).
Start command: `gunicorn server:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4`
Environment variables: `GEMINI_API_KEY`, `ROBOT_TOKEN`
