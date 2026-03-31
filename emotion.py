import cv2
import numpy as np
from deepface import DeepFace

from temporal_model import TemporalEmotionModel
from explainer import EmotionExplainer
from utils import EmotionVisualizer, PerformanceMonitor, EmotionLogger

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LAYOUT & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHOW_HEATMAP           = True
SHOW_REGION_BOXES      = False
SHOW_PANEL             = True
SKIP_FRAMES            = 4
CONFIDENCE_THRESHOLD   = 0.35

NAV_W      = 80       # sidebar icon-strip width
CONTENT_W  = 460      # main content area width
PANEL_W    = NAV_W + CONTENT_W   # total panel width = 540

TAB_ANALYSIS  = 0
TAB_REGIONS   = 1
TAB_TIMELINE  = 2
active_tab    = TAB_ANALYSIS

WIN_NAME = "Facial Emotion Recognition – XAI"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COLOURS  (BGR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EC = {
    'happy':    ( 50, 220,  80),
    'sad':      (200,  80,  40),
    'angry':    ( 40,  40, 240),
    'surprise': (  0, 210, 255),
    'fear':     (200,  70, 210),
    'disgust':  ( 40, 175,  80),
    'neutral':  (170, 170, 170),
}

BG    = ( 16,  18,  24)
NAV   = ( 12,  14,  18)   # sidebar background
CARD  = ( 28,  30,  38)
CARD2 = ( 40,  42,  54)
SEP   = ( 50,  53,  68)
WHITE = (235, 238, 248)
MUTED = (110, 115, 138)
ACC   = (100, 200, 255)   # fallback accent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DRAWING UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fill(img, x1, y1, x2, y2, color):
    if x2 > x1 and y2 > y1:
        img[y1:y2, x1:x2] = color

def grad_bar(img, x1, y1, x2, y2, c_left, c_right):
    """Vectorised horizontal gradient – no Python loop."""
    w = x2 - x1
    if w <= 0 or y2 <= y1:
        return
    t    = np.linspace(0, 1, w, dtype=np.float32)
    grad = ((1 - t)[:, None] * np.array(c_left,  np.float32) +
                  t[:, None]  * np.array(c_right, np.float32)).astype(np.uint8)
    img[y1:y2, x1:x2] = grad[np.newaxis, :, :]

def rr(img, x1, y1, x2, y2, color, r=8):
    """Filled rounded rectangle."""
    r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
    img[y1+r:y2-r, x1:x2]   = color
    img[y1:y2,   x1+r:x2-r] = color
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx,cy), r, color, -1)

def txt(img, s, x, y, scale, color, bold=False, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, s, (x,y), font, scale, color,
                2 if bold else 1, cv2.LINE_AA)

def txd(img, s, x, y, scale, color, bold=False):
    cv2.putText(img, s, (x,y), cv2.FONT_HERSHEY_DUPLEX, scale, color,
                2 if bold else 1, cv2.LINE_AA)

def prob_bar(img, x1, x2, y, bh, prob, color):
    fill(img, x1, y, x2, y+bh, SEP)
    fw = int(prob * (x2-x1))
    if fw > 0:
        dark = tuple(max(0,c-90) for c in color)
        grad_bar(img, x1, y, x1+fw, y+bh, dark, color)

def section_hdr(img, x0, y, w, label, accent, icon=""):
    tint = tuple(c//5 for c in accent)
    rr(img, x0, y, x0+w, y+36, tint, r=6)
    cv2.rectangle(img, (x0, y), (x0+5, y+36), accent, -1)
    txt(img, f"{icon}  {label}", x0+14, y+26, 0.72, WHITE, bold=True)
    return y+50


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LEFT SIDEBAR NAV  (clickable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Each button: label, short icon char, Y-start in the combined image
NAV_LOGO_H  = 90          # height of logo area
NAV_BTN_H   = 100         # height per nav button
NAV_TABS    = [
    ("ANALYSIS",  "A", TAB_ANALYSIS),
    ("REGIONS",   "R", TAB_REGIONS),
    ("TIMELINE",  "T", TAB_TIMELINE),
]

def nav_btn_region(frame_h_unused, idx):
    """Return (y1, y2) of nav button idx in the FULL combined image."""
    y1 = NAV_LOGO_H + idx * NAV_BTN_H
    return y1, y1 + NAV_BTN_H

def draw_sidebar(img, panel_x, height, accent, tab):
    """Draw the left icon sidebar onto 'img' starting at x=panel_x."""
    # Background
    fill(img, panel_x, 0, panel_x + NAV_W, height, NAV)
    # Thin separator line on the right
    cv2.line(img, (panel_x + NAV_W - 1, 0), (panel_x + NAV_W - 1, height),
             (40, 42, 52), 1)

    # Logo / brand area
    fill(img, panel_x, 0, panel_x + NAV_W, NAV_LOGO_H, (18,20,26))
    txd(img, "FER", panel_x + 8, 38, 0.9, accent, bold=True)
    txt(img, "XAI", panel_x + 14, 66, 0.58, MUTED)

    # Tab buttons
    for idx, (label, icon, tid) in enumerate(NAV_TABS):
        y1, y2 = nav_btn_region(height, idx)
        active = (tab == tid)

        # Button background
        bg_c = CARD if active else NAV
        fill(img, panel_x, y1, panel_x + NAV_W, y2, bg_c)

        # Active: left accent bar
        if active:
            cv2.rectangle(img, (panel_x, y1+8), (panel_x+4, y2-8), accent, -1)
        else:
            # hover-look: subtle separator
            cv2.line(img, (panel_x+10, y2-1), (panel_x+NAV_W-10, y2-1), SEP, 1)

        # Icon circle
        cx = panel_x + NAV_W // 2
        cy = y1 + 40
        ic = accent if active else MUTED
        cv2.circle(img, (cx, cy), 22, tuple(c//4 for c in ic), -1)
        cv2.circle(img, (cx, cy), 22, ic, 1)
        txd(img, icon, cx-10, cy+9, 0.82, ic, bold=active)

        # Label
        lc = WHITE if active else MUTED
        tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[:2]
        lx = panel_x + (NAV_W - tw[0]) // 2
        txt(img, label, lx, y1 + 76, 0.4, lc)

    # Bottom: key hint
    txt(img, "q:quit", panel_x + 5, height - 20, 0.38, (60,62,75))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FACE BOX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_face_box(frame, x, y, w, h, emotion, confidence):
    accent = EC.get(emotion, ACC)
    arm    = max(20, min(w // 4, 40))
    t      = 3

    # Corner brackets
    segs = [
        ((x, y+arm),      (x, y),        (x+arm, y)),
        ((x+w-arm, y),    (x+w, y),      (x+w, y+arm)),
        ((x+w, y+h-arm),  (x+w, y+h),    (x+w-arm, y+h)),
        ((x+arm, y+h),    (x, y+h),      (x, y+h-arm)),
    ]
    for p1, corner, p2 in segs:
        cv2.line(frame, p1, corner, accent, t, cv2.LINE_AA)
        cv2.line(frame, corner,  p2, accent, t, cv2.LINE_AA)

    # Subtle inner rect
    ov = frame.copy()
    cv2.rectangle(ov, (x,y), (x+w,y+h), accent, 1)
    cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)

    # Label badge below face
    label = f"  {emotion.upper()}  {int(confidence*100)}%  "
    fs    = 0.78
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
    bx1 = x
    by1 = y + h + 6
    bx2 = x + tw + 4
    by2 = by1 + th + 16
    bg_c = tuple(max(0, c//5) for c in accent)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_c, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), accent, 2)
    txd(frame, label, bx1+2, by2-8, fs, accent, bold=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CAMERA TOP-BAR OVERLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_camera_ui(frame, emotion, confidence, fps):
    fh, fw = frame.shape[:2]
    accent  = EC.get(emotion, ACC)
    bar_h   = 96

    # semi-transparent strip
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (fw, bar_h), (10,12,18), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (0,0), (7, bar_h), accent, -1)

    # Emotion
    txt(frame, "DETECTED EMOTION", 20, 28, 0.62, MUTED)
    txd(frame, emotion.upper(), 20, 82, 1.8, accent, bold=True)

    # Confidence
    badge = (50,220,80) if confidence>0.7 else (0,200,220) if confidence>0.5 else (30,120,255)
    txt(frame, "CONFIDENCE", fw-190, 28, 0.62, MUTED)
    txd(frame, f"{int(confidence*100)}%", fw-185, 82, 1.7, badge, bold=True)

    # Gradient bar
    by = bar_h - 10
    fill(frame, 20, by, fw-20, by+7, SEP)
    fx = 20 + int(confidence * (fw-40))
    grad_bar(frame, 20, by, fx, by+7, tuple(max(0,c-110) for c in accent), accent)

    # FPS
    fps_c = (50,230,80) if fps>=20 else (0,210,220) if fps>=12 else (30,120,255)
    cv2.rectangle(frame, (8, fh-48), (160, fh-8), (10,12,18), -1)
    txt(frame, f"FPS  {fps:.1f}", 16, fh-18, 0.72, fps_c, bold=True)

    return frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONTENT HEADER  (sits to the right of the sidebar)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TAB_TITLES = ["Emotion Analysis", "Facial Regions", "Timeline & Stats"]
TAB_SUBS   = ["Probabilities & confidence", "Region attention map", "Session history"]

def content_header(panel, cw, accent, tab):
    """Draw content-area header. Returns y after header."""
    fill(panel, 0, 0, cw, 90, CARD)
    cv2.rectangle(panel, (0, 86), (cw, 90), accent, -1)      # accent underline

    txd(panel, TAB_TITLES[tab], 16, 42, 0.95, WHITE, bold=True)
    txt(panel, TAB_SUBS[tab],   16, 72, 0.6,  MUTED)

    # LIVE dot
    cv2.circle(panel, (cw-30, 30), 9, (40,210,80), -1)
    cv2.circle(panel, (cw-30, 30), 9, (80,255,120), 2)
    txt(panel, "LIVE", cw-70, 56, 0.48, (80,255,120))

    return 106


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1: ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_analysis(panel, cw, ch, emotion, confidence, all_probs, emotion_labels, accent):
    y = content_header(panel, cw, accent, TAB_ANALYSIS)

    # ── Emotion card ──────────────────────────────────────
    card_h = 120
    rr(panel, 10, y, cw-10, y+card_h, CARD, r=12)
    cv2.rectangle(panel, (10, y+14), (16, y+card_h-14), accent, -1)

    txt(panel, "Detected Emotion", 28, y+32,  0.65, MUTED)
    txd(panel, emotion.upper(),    28, y+94,  1.7,  accent, bold=True)

    # Confidence badge
    conf_pct = int(confidence * 100)
    badge_c  = (50,220,80) if confidence>0.7 else (0,200,220) if confidence>0.5 else (30,120,255)
    tint_b   = tuple(c//5 for c in badge_c)
    rr(panel, cw-128, y+14, cw-14, y+72, tint_b, r=10)
    txd(panel, f"{conf_pct}%", cw-116, y+60, 1.25, badge_c, bold=True)
    txt(panel, "confidence",   cw-116, y+88, 0.5, MUTED)

    # Confidence gradient bar
    bby = y + card_h - 18
    fill(panel, 24, bby, cw-24, bby+12, SEP)
    fx = 24 + int(confidence * (cw-48))
    grad_bar(panel, 24, bby, fx, bby+12, tuple(max(0,c-100) for c in accent), accent)

    y += card_h + 20

    # ── All Emotion Probabilities ─────────────────────────
    y = section_hdr(panel, 10, y, cw-20, "All Emotion Scores", accent, ">")

    if all_probs is not None:
        ps = np.sum(all_probs)
        nrm = all_probs / ps if ps > 0 else all_probs
        sorted_ep = sorted(zip(emotion_labels, nrm), key=lambda x: x[1], reverse=True)

        for lbl, prob in sorted_ep:
            if y > ch - 60:
                break
            is_top = (lbl == emotion)
            ec     = EC.get(lbl, (130,130,160))

            if is_top:
                rr(panel, 10, y-22, cw-10, y+16, tuple(c//6 for c in ec), r=6)

            lc = ec if is_top else MUTED
            wt = True if is_top else False

            txt(panel, lbl.capitalize(),       18, y, 0.72, lc, bold=wt)
            txt(panel, f"{prob*100:.1f}%",     175, y, 0.68, lc, bold=wt)
            prob_bar(panel, 255, cw-18, y-16, 16, prob, ec)

            y += 38

    # Low confidence hint
    if confidence < 0.6 and y < ch-55:
        y += 5
        txt(panel, "Low confidence – see Regions tab for detail", 14, y, 0.52, (120,120,80))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2: REGIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONS = {
    'happy':    ["Mouth corners raised upward", "Eye crinkles (Duchenne smile)", "Cheeks elevated"],
    'sad':      ["Inner brow corners raised", "Mouth corners pulled down", "Eyelids drooping"],
    'angry':    ["Brows furrowed together", "Nostrils flared", "Jaw and lips tightened"],
    'surprise': ["Eyes opened wide", "Brows arched high", "Jaw dropped open"],
    'fear':     ["Eyes wide with tension", "Brows raised & drawn in", "Lips stretched back"],
    'disgust':  ["Upper lip raised / curled", "Nose bridge wrinkled", "Brows pulled low"],
    'neutral':  ["Relaxed facial muscles", "No dominant expression cues", "Balanced feature positions"],
}

def tab_regions(panel, cw, ch, emotion, confidence, accent, explainer_obj):
    y = content_header(panel, cw, accent, TAB_REGIONS)

    txt(panel, "Attention weights show which facial", 14, y,    0.62, MUTED)
    txt(panel, "areas drove this prediction.",         14, y+26, 0.62, MUTED)
    y += 54

    # ── Region importance ─────────────────────────────────
    y = section_hdr(panel, 10, y, cw-20, "Attention Weights", accent, "#")

    rimp   = explainer_obj.region_importance.get(emotion, {})
    sorted_r = sorted(rimp.items(), key=lambda x: x[1], reverse=True)
    pair_c = [accent, (200,170,60), (120,120,150), (80,160,200), (160,80,200)]

    for i, (region, imp) in enumerate(sorted_r[:5]):
        if y > ch - 200:
            break
        rc = pair_c[min(i, len(pair_c)-1)]
        txt(panel,  region.title(),     18, y, 0.7, WHITE, bold=True)
        txt(panel, f"{imp*100:.0f}%",  cw-58, y, 0.68, rc)
        prob_bar(panel, 18, cw-68, y+6, 16, imp, rc)
        y += 46

    y += 4
    # ── Reasoning ─────────────────────────────────────────
    y = section_hdr(panel, 10, y, cw-20, "Model Reasoning", accent, "?")

    for reason in REASONS.get(emotion, ["No detail available."]):
        if y > ch - 45:
            break
        cv2.circle(panel, (24, y-5), 5, accent, -1)
        txt(panel, reason, 38, y, 0.64, (190,195,215))
        y += 32


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3: TIMELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_timeline(panel, cw, ch, emotion, visualizer_obj, accent):
    y = content_header(panel, cw, accent, TAB_TIMELINE)

    # ── Timeline chart ────────────────────────────────────
    y = section_hdr(panel, 10, y, cw-20, "Emotion Over Time", accent, "~")

    tl_h = 130
    tl_w = cw - 28
    tl_img = visualizer_obj.create_timeline_graph(tl_w, tl_h)

    # Dark-background version of chart
    tl_dark = np.full_like(tl_img, CARD2)
    mask    = tl_img.sum(axis=2) < 700
    tl_dark[mask] = tl_img[mask]

    if y + tl_h <= ch:
        panel[y:y+tl_h, 14:14+tl_w] = tl_dark
    y += tl_h + 16

    # ── Session statistics ────────────────────────────────
    y = section_hdr(panel, 10, y, cw-20, "Session Statistics", accent, "%")

    stats = visualizer_obj.get_emotion_statistics()
    if stats:
        for emo, data in sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True):
            if y > ch - 45:
                break
            ec  = EC.get(emo, (130,130,160))
            pct = data['percentage']
            cnt = data['count']

            txt(panel, emo.capitalize(),   18,     y, 0.7, ec, bold=True)
            txt(panel, f"{cnt}x",          172,    y, 0.64, MUTED)
            txt(panel, f"{pct:.1f}%",      cw-72,  y, 0.68, ec)
            prob_bar(panel, 220, cw-85, y-14, 14, pct/100.0, ec)
            y += 36
    else:
        txt(panel, "  No session data yet...", 16, y, 0.65, MUTED)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FULL PANEL BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_panel(frame_w, height, emotion, confidence, all_probs, emotion_labels,
                tab, explainer_obj, visualizer_obj):
    """Return a (height x PANEL_W) panel image."""
    accent  = EC.get(emotion, ACC)
    panel   = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)

    # ── Left sidebar (drawn directly on panel) ────────────
    draw_sidebar(panel, 0, height, accent, tab)

    # ── Content area (slice of panel) ────────────────────
    content = panel[:, NAV_W:]    # view, no copy
    cw, ch  = CONTENT_W, height

    # Fill content background
    fill(content, 0, 0, cw, ch, BG)

    if tab == TAB_ANALYSIS:
        tab_analysis(content, cw, ch, emotion, confidence, all_probs,
                     emotion_labels, accent)
    elif tab == TAB_REGIONS:
        tab_regions(content, cw, ch, emotion, confidence, accent, explainer_obj)
    elif tab == TAB_TIMELINE:
        tab_timeline(content, cw, ch, emotion, visualizer_obj, accent)

    # Keyboard hint bar
    fill(panel, 0, height-28, PANEL_W, height, (10,11,16))
    txt(panel, "1/2/3: switch tab    h: heatmap    r: reset    q: quit",
        8, height-9, 0.42, (60,62,78))

    return panel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MOUSE CALLBACK  – sidebar tab switching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
frame_width_global = 640   # updated after cap.read()

def mouse_cb(event, mx, my, flags, param):
    global active_tab
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # Click must be inside the sidebar x-range
    if not (frame_width_global <= mx <= frame_width_global + NAV_W):
        return
    # Check which nav button was hit
    for idx, (_, _, tid) in enumerate(NAV_TABS):
        y1, y2 = nav_btn_region(0, idx)
        if y1 <= my <= y2:
            active_tab = tid
            break


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INITIALISE COMPONENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
face_cascade        = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
temporal_model      = TemporalEmotionModel(sequence_length=30)
explainer           = EmotionExplainer()
visualizer          = EmotionVisualizer()
performance_monitor = PerformanceMonitor()
logger              = EmotionLogger()

cap          = cv2.VideoCapture(0)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width_global = frame_width

frame_counter   = 0
last_emotion    = None
last_confidence = 0.0
last_probs      = None
last_faces      = []

print("=" * 60)
print("  FER  –  Explainable AI Edition")
print("=" * 60)
print("  Click the sidebar buttons A / R / T to switch tabs.")
print("  Keyboard: 1/2/3 tabs  |  h heatmap  |  r reset  |  q quit")
print()

# Create window and register mouse callback
cv2.namedWindow(WIN_NAME)
cv2.setMouseCallback(WIN_NAME, mouse_cb)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        performance_monitor.update()
        frame_counter += 1

        # Face detection every frame (cheap)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
        if len(faces) > 0:
            last_faces = faces

        # Emotion analysis every SKIP_FRAMES
        if frame_counter % SKIP_FRAMES == 0 and len(last_faces) > 0:
            for (x, y, w, h) in last_faces:
                try:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue
                    res = DeepFace.analyze(roi, actions=['emotion'],
                                           enforce_detection=False, silent=True)
                    temporal_model.update_history(res[0]['emotion'])
                    emo, conf, wp = temporal_model.predict_temporal_emotion()
                    if emo and conf > CONFIDENCE_THRESHOLD:
                        last_emotion    = emo
                        last_confidence = conf
                        last_probs      = wp
                        visualizer.add_emotion(emo)
                        logger.log_emotion(emo, conf)
                        if SHOW_HEATMAP:
                            frame = explainer.apply_heatmap_overlay(
                                frame, roi, emo, x, y, w, h, alpha=0.22)
                except Exception:
                    pass

        # Draw face boxes
        for (x, y, w, h) in last_faces:
            em = last_emotion or "neutral"
            co = last_confidence if last_emotion else 0.0
            draw_face_box(frame, x, y, w, h, em, co)
            if SHOW_REGION_BOXES and last_emotion:
                frame = explainer.draw_region_boxes(frame, last_emotion, x, y, w, h)

        # Camera UI overlay
        if last_emotion:
            frame = draw_camera_ui(frame, last_emotion, last_confidence,
                                   performance_monitor.get_fps())

        # Assemble combined view
        if SHOW_PANEL and last_emotion:
            panel = build_panel(
                frame_width, frame_height,
                last_emotion, last_confidence,
                last_probs, temporal_model.emotion_labels,
                active_tab, explainer, visualizer
            )
            combined = np.hstack([frame, panel])
        else:
            combined = frame

        cv2.imshow(WIN_NAME, combined)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('1'): active_tab = TAB_ANALYSIS
        elif key == ord('2'): active_tab = TAB_REGIONS
        elif key == ord('3'): active_tab = TAB_TIMELINE
        elif key == ord('h'):
            SHOW_HEATMAP = not SHOW_HEATMAP
            print(f"  Heatmap: {'ON' if SHOW_HEATMAP else 'OFF'}")
        elif key == ord('b'):
            SHOW_REGION_BOXES = not SHOW_REGION_BOXES
        elif key == ord('p'):
            SHOW_PANEL = not SHOW_PANEL
        elif key == ord('r'):
            temporal_model.reset()
            visualizer   = EmotionVisualizer()
            last_emotion = None
            print("  Session reset.")
        elif key == ord('s'):
            logger.save_log()

except KeyboardInterrupt:
    print("\n  Stopped.")

finally:
    logger.save_log()
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    stats = visualizer.get_emotion_statistics()
    if stats:
        for emo, data in sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True):
            bar = "█" * int(data['percentage'] / 2)
            print(f"    {emo:10s}: {bar} {data['percentage']:5.1f}%")
    print(f"\n  Average FPS: {performance_monitor.get_fps():.1f}")
    print("=" * 60)
    cap.release()
    cv2.destroyAllWindows()