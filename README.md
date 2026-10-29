# robot-server
University information-desk robot server (Gemini + edge-tts + Google Sheet).
Start command: `gunicorn server:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4`
Environment variables: `GEMINI_API_KEY`, `ROBOT_TOKEN`, `SHEET_CSV_URL` (Google Sheet link; shared "anyone with the link can view")
Sheet layout: see sheet_sample.csv (settings rows: اسم الجامعة / اسم النداء / رسالة الترحيب, then Q&A rows).
