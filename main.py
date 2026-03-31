"""
main.py  –  FER Explainable AI  |  Premium Light Mode  |  PyQt6
Run:  venv/bin/python main.py
"""
import sys, cv2, numpy as np, time
from deepface import DeepFace
from PyQt6.QtWidgets import *
from PyQt6.QtCore import (Qt, QThread, QRectF, QPropertyAnimation, QEasingCurve,
                           pyqtSignal, pyqtProperty, QSize, QTimer, QObject)
from PyQt6.QtGui import *
from temporal_model import TemporalEmotionModel
from explainer import EmotionExplainer
from utils import EmotionVisualizer, PerformanceMonitor, EmotionLogger

# ── Palette ────────────────────────────────────────────────────────────────
SB_BG      = "#1e1b4b"
SB_ACTIVE  = "rgba(129,140,248,0.18)"
SB_ACCENT  = "#818cf8"
SB_TEXT    = "rgba(255,255,255,0.55)"
SB_TEXT_ON = "#ffffff"

PG_BG   = "#f0f1f9"
CARD    = "#ffffff"
BORDER  = "#e4e6f2"
TEXT    = "#111827"
MUTED   = "#6b7280"
ACCENT  = "#6366f1"

EC = {
    "happy":    "#059669",
    "sad":      "#3b82f6",
    "angry":    "#ef4444",
    "surprise": "#f59e0b",
    "fear":     "#8b5cf6",
    "disgust":  "#0d9488",
    "neutral":  "#6b7280",
}
EC_BGR = {
    "happy":    ( 20, 180,  60),
    "sad":      (200,  80,  40),
    "angry":    ( 40,  40, 230),
    "surprise": (  0, 190, 240),
    "fear":     (180,  70, 210),
    "disgust":  ( 40, 160,  80),
    "neutral":  (150, 150, 150),
}
REASONS = {
    "happy":    ["Mouth corners raised upward","Eye crinkles (Duchenne smile)","Cheeks elevated"],
    "sad":      ["Inner brow corners raised","Mouth corners pulled down","Eyelids drooping"],
    "angry":    ["Brows furrowed together","Nostrils flared","Jaw tightened"],
    "surprise": ["Eyes opened wide","Brows arched high","Jaw dropped open"],
    "fear":     ["Eyes wide with tension","Brows raised & drawn inward","Lips stretched back"],
    "disgust":  ["Upper lip raised or curled","Nose bridge wrinkled","Brows pulled low"],
    "neutral":  ["Relaxed facial muscles","No dominant cues","Balanced features"],
}
REGIONS_ALL = ["eyebrows","eyes","nose","mouth","cheeks","face"]
EMOTIONS_ALL = ["happy","sad","angry","surprise","fear","disgust","neutral"]
SKIP = 4
THRESH = 0.35

# ── Helpers ────────────────────────────────────────────────────────────────
def shadow(w, blur=18, dy=4, alpha=22):
    e = QGraphicsDropShadowEffect()
    e.setBlurRadius(blur); e.setOffset(0, dy)
    e.setColor(QColor(30, 30, 80, alpha))
    w.setGraphicsEffect(e)

def lbl(text, size=13, color=TEXT, bold=False, wrap=False):
    l = QLabel(text)
    l.setFont(QFont("SF Pro Display,Helvetica Neue,Arial", size,
                    QFont.Weight.Bold if bold else QFont.Weight.Normal))
    l.setStyleSheet(f"color:{color};background:transparent;")
    if wrap: l.setWordWrap(True)
    return l

def divider():
    d = QFrame(); d.setFrameShape(QFrame.Shape.HLine)
    d.setFixedHeight(1)
    d.setStyleSheet(f"background:{BORDER};border:none;"); return d

# ── Toggle Switch ──────────────────────────────────────────────────────────
class Toggle(QWidget):
    toggled = pyqtSignal(bool)
    def __init__(self, on=False, accent=ACCENT):
        super().__init__(); self._on = on; self._accent = QColor(accent)
        self.setFixedSize(46, 26); self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._x = 22.0 if on else 2.0
        self._anim = QPropertyAnimation(self, b"handle_pos", self)
        self._anim.setDuration(200); self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    @pyqtProperty(float)
    def handle_pos(self): return self._x
    @handle_pos.setter
    def handle_pos(self, v): self._x = v; self.update()

    def isChecked(self): return self._on
    def setChecked(self, v):
        self._on = v
        self._anim.stop()
        self._anim.setStartValue(self._x)
        self._anim.setEndValue(22.0 if v else 2.0)
        self._anim.start()
    def mousePressEvent(self, e):
        self.setChecked(not self._on); self.toggled.emit(self._on)
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height(); r = h/2
        track = self._accent if self._on else QColor("#d1d5db")
        p.setBrush(QBrush(track)); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(0,0,w,h), r, r)
        p.setBrush(QBrush(QColor("#fff")))
        p.drawEllipse(QRectF(self._x, 2, h-4, h-4))

# ── Gradient Bar ───────────────────────────────────────────────────────────
class Bar(QWidget):
    def __init__(self, h=8):
        super().__init__(); self.setFixedHeight(h)
        self._v = 0.0; self._c = QColor(ACCENT)
    def set(self, v, hex_c):
        self._v = max(0.0, min(1.0, v)); self._c = QColor(hex_c); self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height(); r = h/2
        p.setBrush(QBrush(QColor("#e5e7eb"))); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(0,0,w,h), r, r)
        fw = int(self._v * w)
        if fw > 2:
            g = QLinearGradient(0,0,fw,0)
            dark = QColor(max(0,self._c.red()-70),max(0,self._c.green()-70),max(0,self._c.blue()-70))
            g.setColorAt(0, dark); g.setColorAt(1, self._c)
            p.setBrush(QBrush(g))
            p.drawRoundedRect(QRectF(0,0,fw,h), r, r)

# ── Timeline Chart ─────────────────────────────────────────────────────────
class Chart(QWidget):
    def __init__(self):
        super().__init__(); self.setFixedHeight(120); self._h = []
    def set(self, h): self._h = h; self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        path = QPainterPath(); path.addRoundedRect(QRectF(0,0,w,h), 10, 10)
        p.fillPath(path, QColor("#f8f9ff"))
        p.setPen(QColor(BORDER)); p.drawPath(path)
        if not self._h:
            p.setPen(QColor(MUTED))
            p.setFont(QFont("SF Pro Display,Helvetica,Arial", 12))
            p.drawText(QRectF(0,0,w,h), Qt.AlignmentFlag.AlignCenter, "Waiting for data…")
            return
        n = len(self._h); sw = w / max(n, 1); cy = h // 2
        for i, emo in enumerate(self._h):
            x1 = int(i*sw); x2 = int((i+1)*sw)
            c = QColor(EC.get(emo, "#aaa"))
            bg = QColor(c.red()//6+210, c.green()//6+210, c.blue()//6+210)
            p.fillRect(x1,0,x2-x1,h,bg)
            p.fillRect(x1, cy-5, x2-x1, 10, c)
        if self._h:
            last = self._h[-1]
            p.setPen(QColor(EC.get(last,"#aaa")))
            p.setFont(QFont("SF Pro Display,Helvetica,Arial", 10, QFont.Weight.Bold))
            p.drawText(QRectF(w-90,h-22,86,18), Qt.AlignmentFlag.AlignRight, last.upper())

# ── Capture Worker ─────────────────────────────────────────────────────────
class Worker(QThread):
    frame_ready = pyqtSignal(QImage)
    data_ready  = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.show_hm = True; self.show_rb = False
        self._reset  = False; self._run   = True
        self.tm   = TemporalEmotionModel(sequence_length=30)
        self.xp   = EmotionExplainer()
        self.viz  = EmotionVisualizer()
        self.perf = PerformanceMonitor()
        self.log  = EmotionLogger()
    def reset(self): self._reset = True
    def stop(self):  self._run   = False
    def _box(self, fr, x, y, w, h, emo):
        a = EC_BGR.get(emo,(0,200,80)); arm = max(20,min(w//4,42)); t=3
        segs=[((x,y+arm),(x,y),(x+arm,y)),((x+w-arm,y),(x+w,y),(x+w,y+arm)),
              ((x+w,y+h-arm),(x+w,y+h),(x+w-arm,y+h)),((x+arm,y+h),(x,y+h),(x,y+h-arm))]
        for p1,c,p2 in segs:
            cv2.line(fr,p1,c,a,t,cv2.LINE_AA); cv2.line(fr,c,p2,a,t,cv2.LINE_AA)
        ov=fr.copy(); cv2.rectangle(ov,(x,y),(x+w,y+h),a,1)
        cv2.addWeighted(ov,0.15,fr,0.85,0,fr)
    def run(self):
        fc=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        cap=cv2.VideoCapture(0); n=0; lf=[]; le=None; lc=0.0; lw=None
        while self._run:
            if self._reset:
                self.tm.reset(); self.viz=EmotionVisualizer()
                le=None; lc=0.0; lw=None; self._reset=False
            ret,fr=cap.read()
            if not ret: self.msleep(40); continue
            self.perf.update(); n+=1
            g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            faces=cv2.CascadeClassifier.detectMultiScale(fc,g,1.1,5,minSize=(40,40))
            if len(faces)>0: lf=faces
            if n%SKIP==0 and len(lf)>0:
                for (fx,fy,fw,fh) in lf:
                    try:
                        roi=fr[fy:fy+fh,fx:fx+fw]
                        if roi.size==0: continue
                        r=DeepFace.analyze(roi,actions=["emotion"],enforce_detection=False,silent=True)
                        self.tm.update_history(r[0]["emotion"])
                        e,c,w=self.tm.predict_temporal_emotion()
                        if e and c>THRESH:
                            le=e; lc=c; lw=w
                            self.viz.add_emotion(e); self.log.log_emotion(e,c)
                    except: pass
            d=fr.copy()
            for (fx,fy,fw,fh) in lf:
                roi=fr[fy:fy+fh,fx:fx+fw]
                if self.show_hm and le and roi.size>0:
                    d=self.xp.apply_heatmap_overlay(d,roi,le,fx,fy,fw,fh,alpha=0.20)
                if self.show_rb and le:
                    d=self.xp.draw_region_boxes(d,le,fx,fy,fw,fh)
                self._box(d,fx,fy,fw,fh,le or "neutral")
            # Resize for display (faster Qt scaling)
            d=cv2.resize(d,(854,480)); rgb=cv2.cvtColor(d,cv2.COLOR_BGR2RGB)
            h2,w2,ch=rgb.shape
            qi=QImage(rgb.data,w2,h2,ch*w2,QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qi)
            if le and n%3==0:
                pd={}
                if lw is not None:
                    ps=float(np.sum(lw)); nrm=lw/ps if ps>0 else lw
                    pd={lb:round(float(p),4) for lb,p in zip(self.tm.emotion_labels,nrm)}
                ri=self.xp.region_importance.get(le,{})
                st=self.viz.get_emotion_statistics()
                tl=list(self.viz.emotion_timeline)[-80:]
                self.data_ready.emit({
                    "emotion":le,"confidence":round(lc,4),"all_probs":pd,
                    "fps":round(self.perf.get_fps(),1),
                    "region_importance":{k:round(v,4) for k,v in ri.items()},
                    "timeline":tl,
                    "stats":{k:{"count":v["count"],"percentage":round(v["percentage"],1)}
                             for k,v in st.items()},
                })
            self.msleep(16)  # ~60fps cap, DeepFace bottleneck anyway
        cap.release(); self.log.save_log()

# ── Analysis Panel ─────────────────────────────────────────────────────────
class AnalysisPanel(QWidget):
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{PG_BG};")
        root=QVBoxLayout(self); root.setContentsMargins(20,20,20,20); root.setSpacing(16)
        # Emotion card
        card=QFrame(); card.setStyleSheet(f"background:{CARD};border-radius:16px;border:1px solid {BORDER};")
        shadow(card); cl=QVBoxLayout(card); cl.setContentsMargins(24,22,24,22); cl.setSpacing(10)
        top=QHBoxLayout()
        self.emo=QLabel("–"); self.emo.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",42,QFont.Weight.Bold))
        self.emo.setStyleSheet(f"color:{ACCENT};background:transparent;")
        self.badge=QLabel("–%"); self.badge.setFixedSize(88,48); self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",22,QFont.Weight.Bold))
        self.badge.setStyleSheet(f"color:{ACCENT};background:#eef2ff;border-radius:12px;")
        top.addWidget(self.emo); top.addStretch(); top.addWidget(self.badge); cl.addLayout(top)
        cl.addWidget(lbl("Confidence",11,MUTED))
        self.conf_bar=Bar(12); cl.addWidget(self.conf_bar)
        root.addWidget(card)
        # Section label
        sl=lbl("All Emotion Scores",13,TEXT,bold=True); root.addWidget(sl)
        # Prob rows – pre-built
        self._rows={}
        sc=QScrollArea(); sc.setFrameShape(QFrame.Shape.NoFrame); sc.setWidgetResizable(True)
        sc.setStyleSheet(f"background:transparent;")
        cw=QWidget(); cw.setStyleSheet("background:transparent;")
        cl2=QVBoxLayout(cw); cl2.setContentsMargins(0,0,0,0); cl2.setSpacing(6)
        for e in EMOTIONS_ALL:
            rw=QWidget(); rw.setStyleSheet("background:transparent;"); rl=QVBoxLayout(rw)
            rl.setContentsMargins(0,4,0,4); rl.setSpacing(4)
            hr=QHBoxLayout()
            nl=QLabel(e.capitalize()); nl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",14))
            nl.setStyleSheet(f"color:{MUTED};background:transparent;")
            pl=QLabel("0.0%"); pl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",14,QFont.Weight.Bold))
            pl.setStyleSheet(f"color:{MUTED};background:transparent;")
            pl.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            hr.addWidget(nl); hr.addStretch(); hr.addWidget(pl); rl.addLayout(hr)
            b=Bar(10); rl.addWidget(b)
            cl2.addWidget(rw); self._rows[e]=(rw,nl,pl,b)
        sc.setWidget(cw); root.addWidget(sc)

    def update(self, data):
        e=data.get("emotion"); c=data.get("confidence",0.0); probs=data.get("all_probs",{})
        if not e: return
        clr=EC.get(e,ACCENT)
        self.emo.setText(e.upper()); self.emo.setStyleSheet(f"color:{clr};background:transparent;")
        self.badge.setText(f"{int(c*100)}%")
        tint=clr.replace("#",""); r,g,b2=int(tint[0:2],16),int(tint[2:4],16),int(tint[4:6],16)
        bg_tint=f"rgb({min(255,r+180)},{min(255,g+180)},{min(255,b2+180)})"
        self.badge.setStyleSheet(f"color:{clr};background:{bg_tint};border-radius:12px;")
        self.conf_bar.set(c,clr)
        for em,(rw,nl,pl,bar) in self._rows.items():
            prob=probs.get(em,0.0); ec=EC.get(em,MUTED); is_top=(em==e)
            bg=f"background:{'#f5f3ff' if is_top else 'transparent'};border-radius:8px;"
            rw.setStyleSheet(bg)
            clr2=ec if is_top else MUTED; wt="700" if is_top else "400"
            nl.setStyleSheet(f"color:{clr2};background:transparent;font-weight:{wt};")
            pl.setStyleSheet(f"color:{clr2};background:transparent;font-weight:{wt};")
            pl.setText(f"{prob*100:.1f}%"); bar.set(prob,ec)

# ── Regions Panel ──────────────────────────────────────────────────────────
class RegionsPanel(QWidget):
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{PG_BG};")
        root=QVBoxLayout(self); root.setContentsMargins(20,20,20,20); root.setSpacing(14)
        root.addWidget(lbl("Attention Weights",13,TEXT,bold=True))
        self._reg_rows={}
        RCS=["#6366f1","#f59e0b","#10b981","#3b82f6","#ec4899","#8b5cf6"]
        for i,reg in enumerate(REGIONS_ALL):
            rw=QWidget(); rw.setStyleSheet("background:transparent;"); rl=QVBoxLayout(rw)
            rl.setContentsMargins(0,4,0,4); rl.setSpacing(4)
            hr=QHBoxLayout()
            nl=QLabel(reg.title()); nl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",14))
            nl.setStyleSheet(f"color:{TEXT};background:transparent;")
            pl=QLabel("0%"); pl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",14,QFont.Weight.Bold))
            pl.setStyleSheet(f"color:{RCS[i]};background:transparent;")
            pl.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            hr.addWidget(nl); hr.addStretch(); hr.addWidget(pl); rl.addLayout(hr)
            b=Bar(10); b.set(0,RCS[i]); rl.addWidget(b)
            root.addWidget(rw); self._reg_rows[reg]=(rw,nl,pl,b,RCS[i])
        root.addWidget(divider())
        root.addWidget(lbl("Model Reasoning",13,TEXT,bold=True))
        self._reason_lbl=QLabel("–"); self._reason_lbl.setWordWrap(True)
        self._reason_lbl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",13))
        self._reason_lbl.setStyleSheet(f"color:{MUTED};background:transparent;")
        root.addWidget(self._reason_lbl); root.addStretch()

    def update(self, data):
        e=data.get("emotion"); ri=data.get("region_importance",{})
        if not e: return
        for reg,(rw,nl,pl,b,rc) in self._reg_rows.items():
            imp=ri.get(reg,0.0); b.set(imp,rc)
            pl.setText(f"{imp*100:.0f}%")
            rw.setVisible(imp>0.01)
        reasons=REASONS.get(e,[])
        self._reason_lbl.setText("\n".join(f"  •  {r}" for r in reasons))

# ── Timeline Panel ─────────────────────────────────────────────────────────
class TimelinePanel(QWidget):
    def __init__(self):
        super().__init__(); self.setStyleSheet(f"background:{PG_BG};")
        root=QVBoxLayout(self); root.setContentsMargins(20,20,20,20); root.setSpacing(14)
        root.addWidget(lbl("Emotion Over Time",13,TEXT,bold=True))
        self.chart=Chart(); root.addWidget(self.chart)
        root.addWidget(divider())
        root.addWidget(lbl("Session Statistics",13,TEXT,bold=True))
        self._stat_rows={}
        for e in EMOTIONS_ALL:
            rw=QWidget(); rw.setStyleSheet("background:transparent;"); rl=QVBoxLayout(rw)
            rl.setContentsMargins(0,4,0,4); rl.setSpacing(4)
            hr=QHBoxLayout(); ec=EC.get(e,MUTED)
            nl=QLabel(e.capitalize()); nl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",14,QFont.Weight.Bold))
            nl.setStyleSheet(f"color:{ec};background:transparent;")
            cl=QLabel(""); cl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",11))
            cl.setStyleSheet(f"color:{MUTED};background:transparent;")
            pl=QLabel("0%"); pl.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",14,QFont.Weight.Bold))
            pl.setStyleSheet(f"color:{ec};background:transparent;")
            pl.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            hr.addWidget(nl); hr.addWidget(cl); hr.addStretch(); hr.addWidget(pl)
            rl.addLayout(hr); b=Bar(8); b.set(0,ec); rl.addWidget(b)
            root.addWidget(rw); self._stat_rows[e]=(rw,nl,cl,pl,b,ec)
        root.addStretch()

    def update(self, data):
        self.chart.set(data.get("timeline",[]))
        stats=data.get("stats",{})
        for e,(rw,nl,cl,pl,b,ec) in self._stat_rows.items():
            if e in stats:
                s=stats[e]; pct=s["percentage"]; cnt=s["count"]
                # Show count as readable "N detections" subtitle
                cl.setText(f"  {cnt} detection{'s' if cnt!=1 else ''}")
                pl.setText(f"{pct:.1f}%"); b.set(pct/100.0,ec)
                rw.setVisible(True)
            else:
                rw.setVisible(False)

# ── Main Window ────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FER – Explainable AI")
        self.resize(1280,760); self.setMinimumSize(1100,680)
        self.worker=Worker()
        self._build(); self._style()
        self.worker.frame_ready.connect(self._frame)
        self.worker.data_ready.connect(self._data)
        self.worker.start()

    def _build(self):
        cw=QWidget(); cw.setObjectName("root"); self.setCentralWidget(cw)
        ml=QHBoxLayout(cw); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)
        ml.addWidget(self._sidebar())
        # Center + right
        cr=QHBoxLayout(); cr.setContentsMargins(0,0,0,0); cr.setSpacing(0)
        cr.addWidget(self._center(), stretch=1)
        cr.addWidget(self._right())
        crw=QWidget(); crw.setLayout(cr); ml.addWidget(crw, stretch=1)

    # ── Sidebar ────────────────────────────────────────────
    def _sidebar(self):
        sb=QFrame(); sb.setObjectName("sb"); sb.setFixedWidth(220)
        sl=QVBoxLayout(sb); sl.setContentsMargins(0,0,0,0); sl.setSpacing(0)
        # Logo
        logo=QFrame(); logo.setObjectName("logo"); logo.setFixedHeight(72)
        ll=QVBoxLayout(logo); ll.setContentsMargins(20,0,20,0)
        t=QLabel("FER  ·  XAI"); t.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",17,QFont.Weight.Bold))
        t.setStyleSheet("color:#fff;background:transparent;")
        s=QLabel("Explainable AI Edition"); s.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",10))
        s.setStyleSheet(f"color:{SB_TEXT};background:transparent;")
        ll.addSpacing(14); ll.addWidget(t); ll.addWidget(s); ll.addSpacing(14)
        sl.addWidget(logo)
        # Nav section
        sl.addWidget(self._sb_section("NAVIGATION"))
        self._nav_btns=[]
        for i,(icon,label) in enumerate([("◎","Analysis"),("◈","Regions"),("◷","Timeline")]):
            btn=QPushButton(f"  {icon}   {label}"); btn.setObjectName("navbtn")
            btn.setCheckable(True); btn.setChecked(i==0); btn.setFixedHeight(46)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",14))
            btn.clicked.connect(lambda _,idx=i: self._tab(idx))
            self._nav_btns.append(btn); sl.addWidget(btn)
        sl.addSpacing(8)
        # Overlays section
        sl.addWidget(self._sb_section("OVERLAYS"))
        sl.addWidget(self._tog_row("Attention Heatmap", True,  self._hm))
        sl.addWidget(self._tog_row("Region Boxes",      False, self._rb))
        sl.addSpacing(8)
        # Actions
        sl.addWidget(self._sb_section("ACTIONS"))
        for lbl_txt, fn in [("  ↺   Reset Session", self._reset),
                             ("  ⬇   Save Log",      self._save)]:
            btn=QPushButton(lbl_txt); btn.setObjectName("actbtn")
            btn.setFixedHeight(40); btn.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",13))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(fn); sl.addWidget(btn)
        sl.addStretch()
        # FPS at bottom
        self.fps=QLabel("FPS  –"); self.fps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps.setFont(QFont("SF Pro Mono,Menlo,Courier",12))
        self.fps.setStyleSheet(f"color:{SB_TEXT};background:transparent;")
        self.fps.setFixedHeight(36); sl.addWidget(self.fps)
        return sb

    def _sb_section(self, text):
        w=QLabel(f"  {text}"); w.setFixedHeight(28)
        w.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",10,QFont.Weight.Bold))
        w.setStyleSheet(f"color:rgba(255,255,255,0.3);background:transparent;letter-spacing:1px;")
        return w

    def _tog_row(self, label, on, cb):
        w=QWidget(); w.setStyleSheet("background:transparent;"); w.setFixedHeight(46)
        hl=QHBoxLayout(w); hl.setContentsMargins(20,0,16,0)
        hl.addWidget(lbl(label,13,SB_TEXT_ON))
        tg=Toggle(on); tg.toggled.connect(cb); hl.addStretch(); hl.addWidget(tg)
        return w

    # ── Center (video) ─────────────────────────────────────
    def _center(self):
        cw=QWidget(); cw.setObjectName("center")
        cl=QVBoxLayout(cw); cl.setContentsMargins(20,20,20,20); cl.setSpacing(0)
        # header
        hdr=QHBoxLayout()
        self.emo_hdr=QLabel("Detecting…"); self.emo_hdr.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",20,QFont.Weight.Bold))
        self.emo_hdr.setStyleSheet(f"color:{TEXT};background:transparent;")
        dot=QLabel("⬤  LIVE"); dot.setStyleSheet("color:#10b981;font-size:13px;background:transparent;")
        hdr.addWidget(self.emo_hdr); hdr.addStretch(); hdr.addWidget(dot)
        cl.addLayout(hdr); cl.addSpacing(12)
        # Video
        self.vid=QLabel(); self.vid.setObjectName("vid")
        self.vid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vid.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.vid.setMinimumSize(480,320)
        shadow(self.vid,24,6,30); cl.addWidget(self.vid, stretch=1)
        return cw

    # ── Right panel ────────────────────────────────────────
    def _right(self):
        rp=QFrame(); rp.setObjectName("right"); rp.setFixedWidth(400)
        rl=QVBoxLayout(rp); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)
        self.stack=QStackedWidget()
        self.ap=AnalysisPanel(); self.rp2=RegionsPanel(); self.tp=TimelinePanel()
        self.stack.addWidget(self.ap); self.stack.addWidget(self.rp2); self.stack.addWidget(self.tp)
        rl.addWidget(self.stack); return rp

    # ── Stylesheet ─────────────────────────────────────────
    def _style(self):
        self.setStyleSheet(f"""
        QWidget#root, QWidget#center {{ background:{PG_BG}; }}
        QFrame#sb {{ background:{SB_BG}; }}
        QFrame#logo {{ background:{SB_BG}; border-bottom:1px solid rgba(255,255,255,0.08); }}
        QFrame#right {{ background:{CARD}; border-left:1px solid {BORDER}; }}
        QLabel#vid {{
            background:#e8eaf6;
            border-radius:16px;
            border:1px solid {BORDER};
        }}
        QPushButton#navbtn {{
            background:transparent; color:{SB_TEXT};
            border:none; border-radius:0; text-align:left; padding-left:8px;
            font-size:14px;
        }}
        QPushButton#navbtn:checked {{
            background:{SB_ACTIVE}; color:{SB_TEXT_ON};
            border-left:3px solid {SB_ACCENT};
        }}
        QPushButton#navbtn:hover:!checked {{
            background:rgba(255,255,255,0.06); color:{SB_TEXT_ON};
        }}
        QPushButton#actbtn {{
            background:transparent; color:rgba(255,255,255,0.7);
            border:none; border-radius:0; text-align:left; padding-left:8px;
        }}
        QPushButton#actbtn:hover {{
            background:rgba(255,255,255,0.06); color:#fff;
        }}
        QPushButton#actbtn:pressed {{ background:rgba(99,102,241,0.2); }}
        QScrollBar:vertical {{ background:{PG_BG}; width:5px; border-radius:3px; }}
        QScrollBar::handle:vertical {{ background:#d1d5db; border-radius:3px; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)

    # ── Slots ──────────────────────────────────────────────
    def _frame(self, qi):
        p=QPixmap.fromImage(qi).scaled(self.vid.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.vid.setPixmap(p)

    def _data(self, d):
        e=d.get("emotion",""); fps=d.get("fps",0.0)
        clr=EC.get(e,ACCENT)
        if e:
            self.emo_hdr.setText(e.upper())
            self.emo_hdr.setStyleSheet(f"color:{clr};background:transparent;")
        fps_clr="#10b981" if fps>=20 else "#f59e0b" if fps>=12 else "#ef4444"
        self.fps.setText(f"FPS  {fps:.1f}")
        self.fps.setStyleSheet(f"color:{fps_clr};background:transparent;")
        self.ap.update(d); self.rp2.update(d); self.tp.update(d)

    def _tab(self, i):
        self.stack.setCurrentIndex(i)
        for j,b in enumerate(self._nav_btns): b.setChecked(j==i)

    def _hm(self, v):  self.worker.show_hm=v
    def _rb(self, v):  self.worker.show_rb=v
    def _reset(self):  self.worker.reset()
    def _save(self):   self.worker.log.save_log()

    def closeEvent(self, e):
        self.worker.stop(); self.worker.wait(3000); e.accept()

# ── Run ────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    app=QApplication(sys.argv)
    app.setApplicationName("FER – Explainable AI")
    app.setFont(QFont("SF Pro Display,Helvetica Neue,Arial",13))
    w=MainWindow(); w.show()
    sys.exit(app.exec())
