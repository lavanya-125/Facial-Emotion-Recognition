"""
app.py – Flask backend for FER Explainable AI
Runs entirely on your local MacBook.

Routes:
  GET  /              → index.html (opens in browser)
  GET  /video_feed    → MJPEG stream (webcam + overlays)
  GET  /emotion_data  → JSON snapshot polled by the browser
  POST /toggle/heatmap
  POST /toggle/regions
  POST /reset
  POST /save
"""

import cv2
import numpy as np
import threading
import time

from flask import Flask, Response, render_template, jsonify, request

from temporal_model import TemporalEmotionModel
from explainer import EmotionExplainer
from utils import EmotionVisualizer, PerformanceMonitor, EmotionLogger
from deepface import DeepFace

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Shared state (read/written from two threads – protected by lock) ──────────
lock = threading.Lock()
state = {
    "emotion":           None,
    "confidence":        0.0,
    "all_probs":         {},
    "fps":               0.0,
    "region_importance": {},
    "timeline":          [],
    "stats":             {},
    "face_count":        0,
    "show_heatmap":      True,
    "show_regions":      False,
}
output_frame_bytes = None   # latest JPEG as bytes

# ── Accent colours ────────────────────────────────────────────────────────────
EC_BGR = {
    "happy":    ( 50, 220,  80),
    "sad":      (200,  80,  40),
    "angry":    ( 40,  40, 240),
    "surprise": (  0, 210, 255),
    "fear":     (200,  70, 210),
    "disgust":  ( 40, 175,  80),
    "neutral":  (170, 170, 170),
}
EC_HEX = {
    "happy":    "#32dc50",
    "sad":      "#c85028",
    "angry":    "#5050f0",
    "surprise": "#00d2ff",
    "fear":     "#c846d2",
    "disgust":  "#28af50",
    "neutral":  "#aaaaaa",
}

SKIP_FRAMES          = 4
CONFIDENCE_THRESHOLD = 0.35

REASONS = {
    "happy":    ["Mouth corners raised upward", "Eye crinkles (Duchenne smile)", "Cheeks elevated"],
    "sad":      ["Inner brow corners raised", "Mouth corners pulled down", "Eyelids drooping"],
    "angry":    ["Brows furrowed together", "Nostrils flared", "Jaw and lips tightened"],
    "surprise": ["Eyes opened wide", "Brows arched high", "Jaw dropped open"],
    "fear":     ["Eyes wide with tension", "Brows raised & drawn inward", "Lips stretched back"],
    "disgust":  ["Upper lip raised or curled", "Nose bridge wrinkled", "Brows pulled low"],
    "neutral":  ["Relaxed facial muscles", "No dominant expression cues", "Balanced feature positions"],
}

# ── Draw corner-bracket face box ──────────────────────────────────────────────
def draw_face_box(frame, x, y, w, h, emotion, confidence):
    accent = EC_BGR.get(emotion, (0, 255, 255))
    arm    = max(22, min(w // 4, 44))
    t      = 3
    segs = [
        ((x,       y + arm), (x,   y),   (x + arm, y)),
        ((x+w-arm, y),       (x+w, y),   (x+w,     y + arm)),
        ((x+w,     y+h-arm), (x+w, y+h), (x+w-arm, y+h)),
        ((x+arm,   y+h),     (x,   y+h), (x,       y+h-arm)),
    ]
    for p1, corner, p2 in segs:
        cv2.line(frame, p1, corner, accent, t, cv2.LINE_AA)
        cv2.line(frame, corner,  p2, accent, t, cv2.LINE_AA)
    # subtle inner rect glow
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), accent, 1)
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)

# ── Background camera + analysis thread ──────────────────────────────────────
def capture_loop():
    global output_frame_bytes

    face_cascade   = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    temporal_model = TemporalEmotionModel(sequence_length=30)
    explainer      = EmotionExplainer()
    visualizer     = EmotionVisualizer()
    perf           = PerformanceMonitor()
    logger         = EmotionLogger()

    cap           = cv2.VideoCapture(0)
    frame_counter  = 0
    last_faces     = []
    last_emotion   = None
    last_confidence = 0.0

    print("  Camera started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        perf.update()
        frame_counter += 1

        # Face detection – every frame (fast)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) > 0:
            last_faces = faces

        # Emotion analysis – every SKIP_FRAMES
        if frame_counter % SKIP_FRAMES == 0 and len(last_faces) > 0:
            for (fx, fy, fw, fh) in last_faces:
                try:
                    roi = frame[fy:fy+fh, fx:fx+fw]
                    if roi.size == 0:
                        continue
                    res = DeepFace.analyze(roi, actions=["emotion"],
                                           enforce_detection=False, silent=True)
                    temporal_model.update_history(res[0]["emotion"])
                    emo, conf, wp = temporal_model.predict_temporal_emotion()

                    if emo and conf > CONFIDENCE_THRESHOLD:
                        last_emotion    = emo
                        last_confidence = conf

                        probs_dict = {}
                        if wp is not None:
                            ps  = float(np.sum(wp))
                            nrm = wp / ps if ps > 0 else wp
                            probs_dict = {
                                lbl: round(float(p), 4)
                                for lbl, p in zip(temporal_model.emotion_labels, nrm)
                            }

                        visualizer.add_emotion(emo)
                        logger.log_emotion(emo, conf)

                        ri    = explainer.region_importance.get(emo, {})
                        stats = visualizer.get_emotion_statistics()
                        tl    = list(visualizer.emotion_timeline)[-80:]

                        with lock:
                            state.update({
                                "emotion":           emo,
                                "confidence":        round(conf, 4),
                                "all_probs":         probs_dict,
                                "fps":               round(perf.get_fps(), 1),
                                "region_importance": {k: round(v, 4) for k, v in ri.items()},
                                "timeline":          tl,
                                "stats": {
                                    k: {"count": v["count"],
                                        "percentage": round(v["percentage"], 1)}
                                    for k, v in stats.items()
                                },
                                "face_count": len(last_faces),
                            })
                except Exception:
                    pass

        # Build display frame with overlays
        display = frame.copy()

        with lock:
            sh = state["show_heatmap"]
            sr = state["show_regions"]

        for (fx, fy, fw, fh) in last_faces:
            roi   = frame[fy:fy+fh, fx:fx+fw]
            emo_d = last_emotion or "neutral"
            cod   = last_confidence if last_emotion else 0.0

            if sh and last_emotion and roi.size > 0:
                display = explainer.apply_heatmap_overlay(
                    display, roi, last_emotion, fx, fy, fw, fh, alpha=0.22
                )
            if sr and last_emotion:
                display = explainer.draw_region_boxes(display, last_emotion, fx, fy, fw, fh)

            draw_face_box(display, fx, fy, fw, fh, emo_d, cod)

        with lock:
            state["fps"] = round(perf.get_fps(), 1)

        # Encode to JPEG
        ok, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with lock:
                output_frame_bytes = jpg.tobytes()

    cap.release()
    logger.save_log()

# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with lock:
                fb = output_frame_bytes
            if fb is None:
                time.sleep(0.02)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n")
            time.sleep(0.033)   # ~30 fps push rate
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/emotion_data")
def emotion_data():
    with lock:
        data = dict(state)
    data["ec_hex"]  = EC_HEX
    data["reasons"] = REASONS
    return jsonify(data)


@app.route("/toggle/heatmap", methods=["POST"])
def toggle_heatmap():
    with lock:
        state["show_heatmap"] = not state["show_heatmap"]
        val = state["show_heatmap"]
    return jsonify({"show_heatmap": val})


@app.route("/toggle/regions", methods=["POST"])
def toggle_regions():
    with lock:
        state["show_regions"] = not state["show_regions"]
        val = state["show_regions"]
    return jsonify({"show_regions": val})


@app.route("/reset", methods=["POST"])
def reset_session():
    with lock:
        state.update({
            "emotion": None, "confidence": 0.0,
            "all_probs": {}, "timeline": [],
            "stats": {}, "region_importance": {},
        })
    return jsonify({"ok": True})


@app.route("/save", methods=["POST"])
def save_log():
    return jsonify({"ok": True, "message": "Log saved to emotion_log.json"})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print()
    print("=" * 50)
    print("  FER – Explainable AI  |  Running locally")
    print("=" * 50)
    print("  → Open your browser at  http://localhost:5000")
    print("  → Everything stays on your MacBook")
    print("  → Press Ctrl-C to stop")
    print("=" * 50)
    print()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
