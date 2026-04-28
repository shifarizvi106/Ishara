"""
SignPath — Python Practice Mode
--------------------------------
Run:  python3 practice.py
Keys: SPACE = next sign  |  B = back  |  S = toggle spell mode  |  Q = quit

Requires:  mediapipe  opencv-python  scikit-learn
           (all already installed from earlier steps)
"""

import cv2
import pickle
import numpy as np
import mediapipe as mp
import os
import sys
import time
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PKL       = os.path.join(os.path.dirname(__file__), 'sign_model.pkl')
HAND_MODEL      = os.path.join(os.path.expanduser('~'), 'Downloads', 'hand_landmarker.task')
CONF_THRESHOLD  = 0.70   # below this → "No sign detected"
SMOOTH_WINDOW   = 12     # rolling majority vote window
HOLD_FRAMES     = 20     # frames to hold before auto-advance (practice mode)
SPELL_HOLD      = 25     # frames to hold before appending letter (spell mode)

# Neon pink colour (BGR for OpenCV)
PINK       = (138, 43, 226)   # actually magenta-ish in BGR
NEON_PINK  = (180, 50, 255)
HOT_PINK   = (147, 20, 255)
WHITE      = (255, 255, 255)
BLACK      = (0, 0, 0)
DARK       = (30, 20, 28)
GREEN      = (80, 220, 80)
GRAY       = (160, 160, 160)

# ─── Signs ────────────────────────────────────────────────────────────────────
SIGNS = [
    ('A', 'Make a fist, thumb resting on the side.'),
    ('B', 'Four fingers up together, thumb tucked across palm.'),
    ('C', 'Curve your hand into a C shape.'),
    ('D', 'Index finger up, other fingers form a circle.'),
    ('E', 'All fingers curled down, thumb tucked under.'),
    ('F', 'Index finger and thumb touch, other fingers up.'),
    ('G', 'Index finger points sideways, thumb parallel.'),
    ('H', 'Index and middle finger point sideways together.'),
    ('I', 'Pinky finger up, others closed.'),
    ('J', 'Pinky up, trace a J in the air.'),
    ('K', 'Index and middle up and spread, thumb between them.'),
    ('L', 'Index finger up, thumb out — like an L shape.'),
    ('M', 'Three fingers folded over the thumb.'),
    ('N', 'Two fingers folded over the thumb.'),
    ('O', 'All fingers and thumb curve to form an O.'),
    ('P', 'Like K but pointing downward.'),
    ('Q', 'Like G but pointing downward.'),
    ('R', 'Index and middle fingers crossed.'),
    ('S', 'Fist with thumb over fingers.'),
    ('T', 'Thumb tucked between index and middle fingers.'),
    ('U', 'Index and middle fingers up together.'),
    ('V', 'Index and middle fingers up and spread — peace sign.'),
    ('W', 'Index, middle, ring fingers up and spread.'),
    ('X', 'Index finger hooked/bent.'),
    ('Y', 'Thumb and pinky out — shaka sign.'),
    ('Z', 'Index finger traces a Z in the air.'),
    ('0', 'Fingers and thumb form an O shape.'),
    ('1', 'Index finger pointing up.'),
    ('2', 'Index and middle fingers up.'),
    ('3', 'Thumb, index, middle fingers up.'),
    ('4', 'Four fingers up, thumb tucked.'),
    ('5', 'All five fingers spread open.'),
]

# ─── Load model ───────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PKL):
    print(f"❌  Model not found: {MODEL_PKL}")
    print("    Run the training script first!")
    sys.exit(1)

with open(MODEL_PKL, 'rb') as f:
    saved = pickle.load(f)
clf = saved['model']
le  = saved['encoder']
print(f"✓  Model loaded — {len(le.classes_)} classes: {list(le.classes_)}")

# ─── MediaPipe ────────────────────────────────────────────────────────────────
if not os.path.exists(HAND_MODEL):
    print(f"❌  Hand model not found: {HAND_MODEL}")
    print("    It should already be in ~/Downloads from earlier.")
    sys.exit(1)

base_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL)
options   = vision.HandLandmarkerOptions(
    base_options=base_opts,
    num_hands=2,
    min_hand_detection_confidence=0.55,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)
print("✓  MediaPipe loaded")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalize(landmarks):
    pts = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.abs(pts)) + 1e-6
    pts /= scale
    return pts.flatten()

def predict(landmarks):
    x = normalize(landmarks).reshape(1, -1)
    proba = clf.predict_proba(x)[0]
    idx   = np.argmax(proba)
    return le.classes_[idx], proba[idx]

def draw_landmarks(frame, landmark_list):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in landmark_list]

    # Connections
    connections = [
        (0,1),(1,2),(2,3),(3,4),           # thumb
        (0,5),(5,6),(6,7),(7,8),           # index
        (0,9),(9,10),(10,11),(11,12),       # middle
        (0,13),(13,14),(14,15),(15,16),     # ring
        (0,17),(17,18),(18,19),(19,20),     # pinky
        (5,9),(9,13),(13,17),              # palm
    ]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (180, 50, 255), 2, cv2.LINE_AA)

    # Joints — neon pink with glow
    for x, y in pts:
        cv2.circle(frame, (x, y), 8,  (180, 50, 255), -1, cv2.LINE_AA)  # glow
        cv2.circle(frame, (x, y), 4,  (255, 80, 220), -1, cv2.LINE_AA)  # core
        cv2.circle(frame, (x, y), 4,  (255,255,255),   1, cv2.LINE_AA)  # outline

def draw_text(frame, text, pos, scale=0.7, color=WHITE, thickness=2, shadow=True):
    x, y = pos
    font = cv2.FONT_HERSHEY_DUPLEX
    if shadow:
        cv2.putText(frame, text, (x+1, y+1), font, scale, BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_pill(frame, text, pos, color=HOT_PINK, text_color=WHITE):
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x, y = pos
    pad = 10
    cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+pad), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+pad), WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, text_color, 1, cv2.LINE_AA)

def draw_confidence_bar(frame, conf, x, y, w=200, h=14):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (60,40,55), -1, cv2.LINE_AA)
    fill = int(w * conf)
    if fill > 0:
        color = (80,220,80) if conf >= CONF_THRESHOLD else (100,100,200)
        cv2.rectangle(frame, (x, y), (x+fill, y+h), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (180,50,255), 1, cv2.LINE_AA)

def draw_hold_ring(frame, cx, cy, progress, label, radius=30):
    # Background ring
    cv2.circle(frame, (cx, cy), radius, (60,40,55), 4, cv2.LINE_AA)
    # Progress arc
    angle = int(360 * progress)
    if angle > 0:
        cv2.ellipse(frame, (cx,cy), (radius,radius), -90, 0, angle, HOT_PINK, 4, cv2.LINE_AA)
    # Label
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),_ = cv2.getTextSize(label, font, 0.7, 1)
    cv2.putText(frame, label, (cx - tw//2, cy + th//2), font, 0.7, WHITE, 1, cv2.LINE_AA)

# ─── State ────────────────────────────────────────────────────────────────────
current      = 0
mode         = 'practice'   # 'practice' | 'spell'
recent_preds = deque(maxlen=SMOOTH_WINDOW)
hold_count   = 0
spell_text   = ''
spell_cool   = 0
last_result  = None
last_conf    = 0.0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("✓  Camera started — press Q to quit")

frame_ts = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Run MediaPipe
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 33
    results  = detector.detect_for_video(mp_img, frame_ts)

    # Dark overlay panel on left
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (320, h), (20,14,22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    detected_label = None
    detected_conf  = 0.0

    if results.hand_landmarks:
        for hand_lm in results.hand_landmarks:
            draw_landmarks(frame, hand_lm)

        # Predict on primary hand
        label, conf = predict(results.hand_landmarks[0])
        detected_conf = conf

        if conf >= CONF_THRESHOLD:
            recent_preds.append(label)
            # Majority vote
            from collections import Counter
            counts = Counter(recent_preds)
            top_label, top_votes = counts.most_common(1)[0]
            if top_votes >= int(SMOOTH_WINDOW * 0.65):
                detected_label = top_label
                detected_conf  = conf
        else:
            recent_preds.clear()
            hold_count = 0

        last_result = detected_label
        last_conf   = detected_conf
    else:
        recent_preds.clear()
        hold_count = 0
        last_result = None

    if spell_cool > 0:
        spell_cool -= 1

    # ── Draw UI ──────────────────────────────────────────────────────────────

    # Mode badge
    mode_color = HOT_PINK if mode == 'practice' else (180, 120, 50)
    mode_label = '📖 PRACTICE' if mode == 'practice' else '✍  SPELL'
    draw_pill(frame, mode_label, (10, 35), color=mode_color)
    draw_text(frame, 'S: toggle mode   SPACE: next   B: back   Q: quit',
              (10, h-12), scale=0.42, color=GRAY, thickness=1, shadow=False)

    # ── Practice mode ────────────────────────────────────────────────────────
    if mode == 'practice':
        sign_char, sign_desc = SIGNS[current % len(SIGNS)]

        # Big sign letter
        cv2.putText(frame, sign_char, (30, 160),
                    cv2.FONT_HERSHEY_DUPLEX, 5.5, HOT_PINK, 8, cv2.LINE_AA)
        cv2.putText(frame, sign_char, (30, 160),
                    cv2.FONT_HERSHEY_DUPLEX, 5.5, WHITE,    2, cv2.LINE_AA)

        # Counter
        draw_text(frame, f'{current+1} / {len(SIGNS)}', (30, 185), scale=0.5, color=GRAY, thickness=1)

        # Description (word-wrap)
        words = sign_desc.split()
        lines, line = [], ''
        for word in words:
            if len(line + word) > 28:
                lines.append(line.strip())
                line = ''
            line += word + ' '
        lines.append(line.strip())
        for i, ln in enumerate(lines):
            draw_text(frame, ln, (12, 215 + i*22), scale=0.48, color=(200,180,210), thickness=1)

        # Detection result
        if detected_label:
            target = sign_char
            is_correct = detected_label == target

            result_color = GREEN if is_correct else (80, 80, 220)
            draw_text(frame, f'Seeing:  {detected_label}', (12, 320), scale=0.65,
                      color=result_color, thickness=2)
            draw_confidence_bar(frame, detected_conf, 12, 332, w=260)
            draw_text(frame, f'{int(detected_conf*100)}% confident', (12, 358),
                      scale=0.45, color=GRAY, thickness=1)

            if is_correct:
                hold_count += 1
                progress = hold_count / HOLD_FRAMES
                draw_hold_ring(frame, 160, 420, progress, 'Hold!')
                draw_text(frame, f'✓  Correct! Hold it...', (12, 470),
                          scale=0.6, color=GREEN, thickness=2)

                if hold_count >= HOLD_FRAMES:
                    # Flash green border
                    cv2.rectangle(frame, (0,0), (w-1,h-1), GREEN, 6)
                    draw_text(frame, '🎉  NICE!', (w//2 - 60, h//2),
                              scale=1.5, color=GREEN, thickness=3)
                    cv2.imshow('SignPath Practice', frame)
                    cv2.waitKey(700)
                    current = (current + 1) % len(SIGNS)
                    hold_count = 0
                    recent_preds.clear()
            else:
                hold_count = 0
                draw_text(frame, 'Keep going...', (12, 470),
                          scale=0.55, color=(160,120,180), thickness=1)
        else:
            draw_text(frame, 'No sign detected', (12, 320),
                      scale=0.6, color=GRAY, thickness=1)
            draw_text(frame, 'Show your hand to the camera', (12, 348),
                      scale=0.45, color=GRAY, thickness=1)

    # ── Spell mode ───────────────────────────────────────────────────────────
    else:
        draw_text(frame, 'FINGER SPELL', (12, 80), scale=0.7, color=HOT_PINK, thickness=2)
        draw_text(frame, 'Hold a letter to add it', (12, 105),
                  scale=0.45, color=GRAY, thickness=1)
        draw_text(frame, 'SPACE key = space   BACKSPACE = delete', (12, 125),
                  scale=0.4, color=GRAY, thickness=1)

        # Spelled text
        disp = spell_text if spell_text else '_'
        # Wrap at 14 chars per line
        chunks = [disp[i:i+14] for i in range(0, max(len(disp),1), 14)]
        for i, ch in enumerate(chunks[-3:]):  # show last 3 lines
            y_pos = 200 + i * 52
            cv2.putText(frame, ch, (12, y_pos),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, HOT_PINK, 3, cv2.LINE_AA)
            cv2.putText(frame, ch, (12, y_pos),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, WHITE,    1, cv2.LINE_AA)

        # Current detection + hold ring
        if detected_label and spell_cool == 0:
            hold_count += 1
            progress = hold_count / SPELL_HOLD
            draw_hold_ring(frame, 160, 390, progress, detected_label, radius=35)
            draw_text(frame, f'Signing: {detected_label}', (12, 440),
                      scale=0.65, color=HOT_PINK, thickness=2)
            draw_confidence_bar(frame, detected_conf, 12, 452, w=260)

            if hold_count >= SPELL_HOLD:
                spell_text += detected_label
                hold_count  = 0
                spell_cool  = 30
                recent_preds.clear()
                # Flash
                cv2.rectangle(frame, (0,0), (w-1,h-1), HOT_PINK, 4)
        elif spell_cool > 0:
            hold_count = 0
            draw_text(frame, '✓ Added!', (12, 440), scale=0.65, color=GREEN, thickness=2)
        else:
            hold_count = 0
            draw_text(frame, 'No sign detected' if not detected_label else f'Seeing: {detected_label}',
                      (12, 440), scale=0.6,
                      color=GRAY if not detected_label else (160,160,220), thickness=1)

    cv2.imshow('SignPath Practice', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord(' '):
        if mode == 'practice':
            current = (current + 1) % len(SIGNS)
        else:
            spell_text += ' '
        hold_count = 0
        recent_preds.clear()
    elif key == ord('b') or key == 8:  # B or backspace
        if mode == 'practice':
            current = (current - 1) % len(SIGNS)
        else:
            spell_text = spell_text[:-1]
        hold_count = 0
        recent_preds.clear()
    elif key == ord('s'):
        mode = 'spell' if mode == 'practice' else 'practice'
        hold_count = 0
        recent_preds.clear()
    elif key == ord('c') and mode == 'spell':
        spell_text = ''

cap.release()
cv2.destroyAllWindows()
print("Bye! 👋")
