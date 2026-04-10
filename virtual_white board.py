import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

board = np.zeros((480, 640, 3), dtype=np.uint8)

COLORS = {
    'Cyan':   (0, 255, 200),
    'Red':    (50, 50, 255),
    'Green':  (0, 230, 80),
    'Yellow': (0, 220, 255),
    'Purple': (220, 80, 255),
    'White':  (255, 255, 255),
}
color_names = list(COLORS.keys())
color_idx = 0
brush = 6

# Smoothing buffer for fingertip
smooth_pts = deque(maxlen=6)

# Wave detection
wrist_x_history = deque(maxlen=20)
wave_timestamps = deque(maxlen=6)
last_wave_dir = None
wave_count = 0
last_erase_time = 0

# Undo history
undo_stack = []

prev_x, prev_y = 0, 0

def fingers_up(lm):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    up = []
    for t, p in zip(tips, pips):
        up.append(lm[t].y < lm[p].y)
    thumb = lm[4].x < lm[3].x
    return thumb, up[0], up[1], up[2], up[3]

def is_fist(lm):
    _, i, m, r, p = fingers_up(lm)
    return not i and not m and not r and not p

def detect_wave(lm, w):
    global last_wave_dir, wave_count, last_erase_time
    wx = int(lm[0].x * w)
    wrist_x_history.append(wx)
    if len(wrist_x_history) < 10:
        return False
    now = time.time()
    if now - last_erase_time < 1.5:
        return False
    recent = list(wrist_x_history)[-10:]
    diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    dir_changes = 0
    current_dir = None
    for d in diffs:
        if abs(d) < 4:
            continue
        new_dir = 'R' if d > 0 else 'L'
        if current_dir and new_dir != current_dir:
            dir_changes += 1
        current_dir = new_dir
    total_motion = max(recent) - min(recent)
    if dir_changes >= 2 and total_motion > 80:
        last_erase_time = now
        wrist_x_history.clear()
        return True
    return False

def get_smooth_point(x, y):
    smooth_pts.append((x, y))
    sx = int(sum(p[0] for p in smooth_pts) / len(smooth_pts))
    sy = int(sum(p[1] for p in smooth_pts) / len(smooth_pts))
    return sx, sy

def draw_ui(frame, mode, color_name, brush_size, wave_detected):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar background
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Color swatches
    for i, (name, col) in enumerate(COLORS.items()):
        cx = 20 + i * 42
        cy = 20
        cv2.circle(frame, (cx, cy), 14, col, -1)
        if i == color_idx:
            cv2.circle(frame, (cx, cy), 14, (255,255,255), 2)
        cv2.putText(frame, str(i+1), (cx-5, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # Brush size display
    bx = 20 + len(COLORS) * 42 + 10
    cv2.circle(frame, (bx + 20, 20), brush_size // 2 + 3, COLORS[color_names[color_idx]], -1)

    # Mode label
    mode_color = (0, 255, 150) if mode == 'DRAWING' else \
                 (0, 180, 255) if mode == 'PAUSE' else \
                 (100, 100, 100)
    cv2.rectangle(frame, (0, 45), (w, 70), (30,30,30), -1)
    cv2.putText(frame, f"Mode: {mode}  |  Color: {color_name}  |  Brush: {brush_size}  |  Wave hand 'bye' to ERASE",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)

    # Wave erase flash
    if wave_detected:
        cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 6)
        cv2.putText(frame, "ERASED!", (w//2 - 70, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 4)

    return frame

print("\n=== Virtual Air Whiteboard ===")
print("Controls:")
print("  ☝  Index finger only  → DRAW")
print("  ✌  Two fingers up     → LIFT PEN / PAUSE")
print("  ✊  Fist               → STOP DRAWING")
print("  👋  Wave bye           → ERASE ALL")
print("  Keys: 1-6=color | +/-=brush | U=undo | C=clear | Q=quit")
print("=====================================\n")

wave_flash_timer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mode = 'IDLE'
    wave_detected = (time.time() - wave_flash_timer < 0.6)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark

        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            result.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(80,80,80), thickness=1, circle_radius=2),
            mp_draw.DrawingSpec(color=(50,50,50), thickness=1)
        )

        ix = int(lm[8].x * w)
        iy = int(lm[8].y * h)
        sx, sy = get_smooth_point(ix, iy)

        _, index_up, middle_up, ring_up, pinky_up = fingers_up(lm)
        fist = is_fist(lm)

        # Wave detection
        if detect_wave(lm, w):
            undo_stack.append(board.copy())
            board = np.zeros((h, w, 3), dtype=np.uint8)
            wave_flash_timer = time.time()
            wave_detected = True
            prev_x, prev_y = 0, 0
            smooth_pts.clear()

        if fist:
            mode = 'FIST'
            prev_x, prev_y = 0, 0
            smooth_pts.clear()

        elif index_up and not middle_up and not ring_up:
            mode = 'DRAWING'
            color = COLORS[color_names[color_idx]]

            if prev_x and prev_y:
                cv2.line(board, (prev_x, prev_y), (sx, sy), color, brush)
                # Extra smoothness: draw circle at each point
                cv2.circle(board, (sx, sy), brush // 2, color, -1)
            prev_x, prev_y = sx, sy

            # Cursor dot
            cv2.circle(frame, (sx, sy), brush // 2 + 4, color, -1)
            cv2.circle(frame, (sx, sy), brush // 2 + 4, (255,255,255), 1)

        elif index_up and middle_up:
            mode = 'PAUSE'
            prev_x, prev_y = 0, 0
            smooth_pts.clear()
            # Show cursor outline only
            cv2.circle(frame, (sx, sy), 12, (150,150,150), 2)

        else:
            mode = 'IDLE'
            prev_x, prev_y = 0, 0
            smooth_pts.clear()

    else:
        prev_x, prev_y = 0, 0
        smooth_pts.clear()
        wrist_x_history.clear()

    # Blend drawing board onto frame
    board_mask = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(board_mask, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    board_fg = cv2.bitwise_and(board, board, mask=mask)
    output = cv2.add(frame_bg, board_fg)

    output = draw_ui(output, mode, color_names[color_idx], brush, wave_detected)

    cv2.imshow("Virtual Air Whiteboard", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        undo_stack.append(board.copy())
        board = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('u') and undo_stack:
        board = undo_stack.pop()
    elif key == ord('+') or key == ord('='):
        brush = min(brush + 2, 40)
    elif key == ord('-'):
        brush = max(brush - 2, 2)
    elif key in [ord(str(i)) for i in range(1, 7)]:
        color_idx = int(chr(key)) - 1

cap.release()
cv2.destroyAllWindows()
