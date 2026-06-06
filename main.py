# import cv2
# import time
# import joblib
# import numpy as np
# from ultralytics import YOLO

# # ─────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────
# MODEL_PATH          = 'yolov8n.pt'
# EXTERNAL_MODEL_PATH = 'traffic_model.pkl'

# ROAD_CONDITION  = 1        # 0: Good  |  1: Moderate  |  2: Bad
# LANE_COUNT      = 2
# LANE_WIDTH_M    = 3.5

# MAX_GREEN = 90
# MIN_GREEN = 10

# PX_PER_M        = 80
# TOTAL_WIDTH_PX  = int(LANE_COUNT * LANE_WIDTH_M * PX_PER_M)
# LANE_ROI        = [200, 480, 0, TOTAL_WIDTH_PX]   # [y1, y2, x1, x2]


# # ─────────────────────────────────────────
# #  HELPER: clip value to a range
# # ─────────────────────────────────────────
# def clip(val, min_v, max_v):
#     return min(max(val, min_v), max_v)


# # ─────────────────────────────────────────
# #  HELPER: time-of-day & rush-hour context
# # ─────────────────────────────────────────
# def get_environmental_context():
#     current_hour = time.localtime().tm_hour

#     if   6  <= current_hour < 12: tod = 0   # Morning
#     elif 12 <= current_hour < 17: tod = 1   # Afternoon
#     elif 17 <= current_hour < 21: tod = 2   # Evening
#     else:                         tod = 3   # Night

#     rush = 1 if (8 <= current_hour <= 10 or 17 <= current_hour <= 19) else 0
#     return tod, rush


# # ─────────────────────────────────────────
# #  CLASS: TrafficBrain — ML prediction
# # ─────────────────────────────────────────
# class TrafficBrain:
#     def __init__(self):
#         try:
#             self.model = joblib.load(EXTERNAL_MODEL_PATH)
#             print("[Brain] ✅ ML model loaded successfully.")
#         except Exception as e:
#             print(f"[Brain] ⚠️  ML model not found ({e}). Using fallback (20s).")
#             self.model = None

#     def predict(self, cars, bikes, trucks):
#         tod, rush = get_environmental_context()

#         # Feature vector must match training-time format exactly
#         input_data = np.array([[cars, bikes, trucks, tod, rush, ROAD_CONDITION, LANE_COUNT]])

#         if self.model:
#             pred = self.model.predict(input_data)[0]
#             return clip(pred, MIN_GREEN, MAX_GREEN)
#         else:
#             # Simple rule-based fallback when no model is available
#             total = cars + bikes + (trucks * 2)   # trucks weigh more
#             fallback = clip(10 + total * 2, MIN_GREEN, MAX_GREEN)
#             return fallback


# # ─────────────────────────────────────────
# #  FUNCTION: load image from disk
# # ─────────────────────────────────────────
# def load_image(image_path: str):
#     """
#     Camera ki jagah: ek image file load karo disk se.
#     Returns the BGR frame (same format as cv2 camera frame).
#     """
#     frame = cv2.imread(image_path)
#     if frame is None:
#         raise FileNotFoundError(
#             f"[Image Loader] ❌ Image not found at: '{image_path}'\n"
#             "Check the path and try again."
#         )
#     print(f"[Image Loader] ✅ Image loaded: {image_path}  |  Size: {frame.shape[1]}×{frame.shape[0]} px")
#     return frame


# # ─────────────────────────────────────────
# #  FUNCTION: perception — count vehicles
# # ─────────────────────────────────────────
# def perception_layer(yolo, frame):
#     y1, y2, x1, x2 = LANE_ROI

#     # Guard: if image is smaller than ROI, use full image
#     h, w = frame.shape[:2]
#     y2 = min(y2, h)
#     x2 = min(x2, w)

#     roi = frame[y1:y2, x1:x2]
#     results = yolo.predict(roi, conf=0.4, verbose=False)

#     cars   = len([b for b in results[0].boxes if int(b.cls) == 2])
#     bikes  = len([b for b in results[0].boxes if int(b.cls) == 3])
#     trucks = len([b for b in results[0].boxes if int(b.cls) == 7])

#     return cars, bikes, trucks


# # ─────────────────────────────────────────
# #  FUNCTION: draw detections on image
# # ─────────────────────────────────────────
# def draw_detections(frame, cars, bikes, trucks, green_time):
#     """Optional: annotate and save the image with results."""
#     y1, y2, x1, x2 = LANE_ROI
#     h, w = frame.shape[:2]
#     y2 = min(y2, h);  x2 = min(x2, w)

#     annotated = frame.copy()

#     # Draw ROI box
#     cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
#     cv2.putText(annotated, "Detection Zone", (x1 + 5, y1 + 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#     # Stats overlay
#     stats = [
#         f"Cars:   {cars}",
#         f"Bikes:  {bikes}",
#         f"Trucks: {trucks}",
#         f"Green:  {green_time:.1f}s",
#     ]
#     for i, txt in enumerate(stats):
#         cv2.putText(annotated, txt, (10, 30 + i * 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     output_path = "annotated_output.jpg"
#     cv2.imwrite(output_path, annotated)
#     print(f"[Output] 🖼️  Annotated image saved → {output_path}")
#     return annotated


# # ─────────────────────────────────────────
# #  MAIN — Sab kuch yahan jodta hai
# # ─────────────────────────────────────────
# def main():
#     print("=" * 50)
#     print("  🚦 AI Traffic Signal System — Image Mode")
#     print("=" * 50)

#     # ── Step 0: Image path input ──────────────────────
#     image_path = input("\n📂 Image ka path enter karo (e.g. traffic.jpg): ").strip()
#     if not image_path:
#         image_path = "traffic.jpg"   # default fallback

#     # ── Step 1: Load models ───────────────────────────
#     print("\n[Setup] Loading YOLO model...")
#     yolo  = YOLO(MODEL_PATH)
#     brain = TrafficBrain()

#     # ── Step 2: Load image (camera ki jagah) ──────────
#     print("\n[Setup] Loading image...")
#     frame = load_image(image_path)

#     # ── Step 3: Detect vehicles ───────────────────────
#     print("\n[Detection] Running YOLO on image...")
#     cars, bikes, trucks = perception_layer(yolo, frame)
#     print(f"[Detection] 🚗 Cars: {cars}  🏍️ Bikes: {bikes}  🚚 Trucks: {trucks}")

#     # ── Step 4: Predict green time ────────────────────
#     print("\n[Brain] Predicting optimal green time...")
#     green_time = brain.predict(cars, bikes, trucks)
#     tod_label  = ["Morning", "Afternoon", "Evening", "Night"]
#     tod, rush  = get_environmental_context()
#     print(f"[Brain] Time of Day : {tod_label[tod]}")
#     print(f"[Brain] Rush Hour   : {'Yes ⚡' if rush else 'No'}")
#     print(f"[Brain] Road Cond.  : {['Good','Moderate','Bad'][ROAD_CONDITION]}")
#     print(f"[Brain] ✅ Predicted Green Time: {green_time:.1f}s")

#     # ── Step 5: Save annotated output image ───────────
#     draw_detections(frame, cars, bikes, trucks, green_time)

#     # ── Step 6: Simulate signal cycle ────────────────
#     print("\n" + "─" * 40)
#     print(f"🟢 GREEN  — {green_time:.1f}s")
#     time.sleep(green_time)

#     print("🟡 YELLOW — 3s")
#     time.sleep(3)

#     print("🔴 RED    — 10s")
#     time.sleep(10)

#     print("\n✅ Signal cycle complete.")
#     print("=" * 50)


# if __name__ == "__main__":
#     main()


import cv2
import time
import joblib
import numpy as np
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Model options (in order of accuracy):
#   'yolov8n.pt'  → nano   (fastest,  least accurate)
#   'yolov8s.pt'  → small  (good balance)          ← recommended
#   'yolov8m.pt'  → medium (better accuracy)
#   'yolov8l.pt'  → large  (best for aerial/dense)
MODEL_PATH          = 'yolov8l.pt'       # ✅ Upgraded from nano → small
EXTERNAL_MODEL_PATH = 'traffic_model.pkl'

ROAD_CONDITION = 1        # 0: Good  |  1: Moderate  |  2: Bad
LANE_COUNT     = 2
LANE_WIDTH_M   = 3.5

MAX_GREEN = 270
MIN_GREEN = 10

# ── Detection tuning ──────────────────────────────────────────
CONF_THRESHOLD = 0.20       # Low threshold — aerial views have lower confidence
IOU_THRESHOLD  = 0.35       # Lower IoU → keeps more overlapping boxes (dense traffic)
IMAGE_SIZE     = 1280       # Larger input size → better small object detection
                            # Default is 640. Use 1280 for dense/aerial scenes.

# ── YOLO COCO class IDs ───────────────────────────────────────
VEHICLE_CLASSES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}
CAR_CLASSES   = [2]
BIKE_CLASSES  = [1, 3]
TRUCK_CLASSES = [5, 7]

# ── ROI: percentage-based (works for any resolution) ──────────
ROI_TOP    = 0.08     # Skip top 8%  (sky / far background)
ROI_BOTTOM = 1.00
ROI_LEFT   = 0.00
ROI_RIGHT  = 1.00

# ── Tiled inference ───────────────────────────────────────────
# Splits image into overlapping tiles — massively improves
# detection in aerial / congested views.
USE_TILING   = True
TILE_ROWS    = 3
TILE_COLS    = 2
TILE_OVERLAP = 0.25      # 25% overlap between tiles


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def clip(val, lo, hi):
    return min(max(val, lo), hi)


def get_environmental_context():
    hour = time.localtime().tm_hour
    if   6  <= hour < 12: tod = 0
    elif 12 <= hour < 17: tod = 1
    elif 17 <= hour < 21: tod = 2
    else:                 tod = 3
    rush = 1 if (8 <= hour <= 10 or 17 <= hour <= 19) else 0
    return tod, rush


def get_tod_label():
    return ["Morning", "Afternoon", "Evening", "Night"][get_environmental_context()[0]]


def get_roi(frame):
    h, w = frame.shape[:2]
    return (int(h * ROI_TOP),  int(h * ROI_BOTTOM),
            int(w * ROI_LEFT), int(w * ROI_RIGHT))


# ═══════════════════════════════════════════════════════════════
#  NMS ACROSS TILES
# ═══════════════════════════════════════════════════════════════

def nms_across_tiles(all_boxes, iou_thresh=0.40):
    """
    Removes duplicate detections that arise from overlapping tiles.
    all_boxes: list of [x1, y1, x2, y2, conf, cls_id]
    """
    if not all_boxes:
        return []

    boxes   = np.array(all_boxes, dtype=np.float32)
    x1, y1  = boxes[:,0], boxes[:,1]
    x2, y2  = boxes[:,2], boxes[:,3]
    scores  = boxes[:,4]
    classes = boxes[:,5]
    areas   = (x2 - x1) * (y2 - y1)
    order   = scores.argsort()[::-1]
    keep    = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = areas[i] + areas[order[1:]] - inter
        iou   = inter / (union + 1e-6)

        same_class = (classes[order[1:]] == classes[i])
        keep_mask  = ~(same_class & (iou > iou_thresh))
        order      = order[np.where(keep_mask)[0] + 1]

    return [all_boxes[k] for k in keep]


# ═══════════════════════════════════════════════════════════════
#  TILED INFERENCE
# ═══════════════════════════════════════════════════════════════

def detect_with_tiling(yolo, roi_frame):
    h, w = roi_frame.shape[:2]
    all_detections = []

    stride_h = int(h / TILE_ROWS)
    stride_w = int(w / TILE_COLS)
    tile_h   = int(stride_h * (1 + TILE_OVERLAP))
    tile_w   = int(stride_w * (1 + TILE_OVERLAP))

    tile_count = 0
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            ty1 = row * stride_h
            tx1 = col * stride_w
            ty2 = min(ty1 + tile_h, h)
            tx2 = min(tx1 + tile_w, w)

            tile = roi_frame[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue

            results = yolo.predict(
                tile,
                conf    = CONF_THRESHOLD,
                iou     = IOU_THRESHOLD,
                imgsz   = IMAGE_SIZE,
                verbose = False,
            )

            for box in results[0].boxes:
                cls_id = int(box.cls)
                if cls_id not in VEHICLE_CLASSES:
                    continue
                bx1, by1, bx2, by2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                all_detections.append([
                    tx1 + bx1, ty1 + by1,
                    tx1 + bx2, ty1 + by2,
                    conf, float(cls_id)
                ])
            tile_count += 1

    print(f"[Detection] Tiles: {tile_count}  |  Raw detections: {len(all_detections)}")
    filtered = nms_across_tiles(all_detections, iou_thresh=0.40)
    print(f"[Detection] After NMS: {len(filtered)} unique vehicles")
    return filtered


# ═══════════════════════════════════════════════════════════════
#  SINGLE-PASS INFERENCE
# ═══════════════════════════════════════════════════════════════

def detect_single_pass(yolo, roi_frame):
    results = yolo.predict(
        roi_frame,
        conf    = CONF_THRESHOLD,
        iou     = IOU_THRESHOLD,
        imgsz   = IMAGE_SIZE,
        verbose = False,
    )
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in VEHICLE_CLASSES:
            continue
        bx1, by1, bx2, by2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append([bx1, by1, bx2, by2, conf, float(cls_id)])
    return detections


# ═══════════════════════════════════════════════════════════════
#  PERCEPTION LAYER
# ═══════════════════════════════════════════════════════════════

def perception_layer(yolo, frame):
    y1, y2, x1, x2 = get_roi(frame)
    roi = frame[y1:y2, x1:x2]
    print(f"[Detection] ROI → y:{y1}–{y2}, x:{x1}–{x2}  ({x2-x1}×{y2-y1} px)")

    if USE_TILING:
        print("[Detection] Mode: Tiled inference")
        detections = detect_with_tiling(yolo, roi)
    else:
        print("[Detection] Mode: Single-pass inference")
        detections = detect_single_pass(yolo, roi)

    cars   = sum(1 for d in detections if int(d[5]) in CAR_CLASSES)
    bikes  = sum(1 for d in detections if int(d[5]) in BIKE_CLASSES)
    trucks = sum(1 for d in detections if int(d[5]) in TRUCK_CLASSES)

    breakdown = {}
    for d in detections:
        name = VEHICLE_CLASSES.get(int(d[5]), 'unknown')
        breakdown[name] = breakdown.get(name, 0) + 1
    print(f"[Detection] Breakdown: {breakdown}")

    return cars, bikes, trucks, detections, (y1, x1)


# ═══════════════════════════════════════════════════════════════
#  TRAFFIC BRAIN
# ═══════════════════════════════════════════════════════════════

class TrafficBrain:
    def __init__(self):
        try:
            self.model    = joblib.load(EXTERNAL_MODEL_PATH)
            self.using_ml = True
            print("[Brain] ✅ ML model loaded.")
        except Exception as e:
            print(f"[Brain] ⚠️  ML model not found ({e}). Using rule-based fallback.")
            self.model    = None
            self.using_ml = False

    def predict(self, cars, bikes, trucks):
        tod, rush = get_environmental_context()
        features  = np.array([[cars, bikes, trucks, tod, rush, ROAD_CONDITION, LANE_COUNT]])

        if self.using_ml:
            try:
                pred = self.model.predict(features)[0]
                return float(clip(pred, MIN_GREEN, MAX_GREEN))
            except Exception as e:
                print(f"[Brain] ⚠️  ML prediction failed: {e}. Using fallback.")

        # Rule-based fallback
        weighted = cars + (bikes * 0.5) + (trucks * 2.5)
        if rush:
            weighted *= 1.2
        return float(clip(MIN_GREEN + weighted * 2, MIN_GREEN, MAX_GREEN))


# ═══════════════════════════════════════════════════════════════
#  IMAGE LOADER
# ═══════════════════════════════════════════════════════════════

def load_image(path: str):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"[Image Loader] ❌ Not found: '{path}'")
    h, w = frame.shape[:2]
    print(f"[Image Loader] ✅ {path}  ({w}×{h} px)")
    return frame


# ═══════════════════════════════════════════════════════════════
#  DRAW DETECTIONS & SAVE
# ═══════════════════════════════════════════════════════════════

def draw_detections(frame, cars, bikes, trucks, green_time, detections, roi_offset):
    annotated = frame.copy()
    oy, ox    = roi_offset
    y1, y2, x1, x2 = get_roi(frame)

    COLORS = {
        'car':        (0,   255,  80),
        'bicycle':    (0,   200, 255),
        'motorcycle': (255, 140,   0),
        'bus':        (80,   80, 255),
        'truck':      (200,   0, 200),
    }

    for det in detections:
        bx1, by1, bx2, by2, conf, cls_id = det
        cls_name = VEHICLE_CLASSES.get(int(cls_id), 'vehicle')
        color    = COLORS.get(cls_name, (255, 255, 255))
        fx1, fy1 = int(ox + bx1), int(oy + by1)
        fx2, fy2 = int(ox + bx2), int(oy + by2)

        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), color, 2)
        label = f"{cls_name} {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(annotated, (fx1, fy1 - lh - 5), (fx1 + lw + 4, fy1), color, -1)
        cv2.putText(annotated, label, (fx1 + 2, fy1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    # ROI border
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(annotated, "Detection Zone", (x1 + 8, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    # Stats panel
    total = cars + bikes + trucks
    panel = [
        ("  AI TRAFFIC SYSTEM  ",    (0, 200, 255)),
        (f"  Cars         : {cars:>3}  ", (255, 255, 255)),
        (f"  Bikes        : {bikes:>3}  ", (255, 255, 255)),
        (f"  Buses/Trucks : {trucks:>3}  ", (255, 255, 255)),
        (f"  TOTAL        : {total:>3}  ", (200, 200, 100)),
        (f"  Green Time   : {green_time:.1f}s  ", (0, 255, 100)),
    ]
    lh, pw = 28, 215
    ph     = lh * len(panel) + 10
    ov     = annotated.copy()
    cv2.rectangle(ov, (10, 10), (10 + pw, 10 + ph), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, annotated, 0.4, 0, annotated)
    for i, (txt, col) in enumerate(panel):
        cv2.putText(annotated, txt, (15, 32 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)

    out = "annotated_output.jpg"
    cv2.imwrite(out, annotated)
    print(f"[Output] 🖼️  Saved → {out}")


# ═══════════════════════════════════════════════════════════════
#  SIGNAL SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_signal_cycle(green_time: float):
    print("\n" + "─" * 45)
    print(f"  🟢  GREEN  →  {green_time:.1f} seconds")
    time.sleep(min(green_time, 5))   # Capped for demo; remove cap for real use

    # print(f"  🟡  YELLOW →  3 seconds")
    # time.sleep(3)

    # print(f"  🔴  RED    →  10 seconds")
    # time.sleep(3)                    # Shortened for demo

    print("─" * 45)
    print("  ✅  Cycle complete.\n")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(cars, bikes, trucks, green_time, using_ml):
    tod, rush  = get_environmental_context()
    road_label = ['Good ✅', 'Moderate ⚠️', 'Bad ❌'][ROAD_CONDITION]
    print("\n" + "═" * 45)
    print("  📊  DETECTION SUMMARY")
    print("═" * 45)
    print(f"  🚗  Cars          : {cars}")
    print(f"  🏍️  Bikes         : {bikes}")
    print(f"  🚌  Buses/Trucks  : {trucks}")
    print(f"  🔢  Total         : {cars + bikes + trucks}")
    print("─" * 45)
    print(f"  🕐  Time of Day   : {get_tod_label()}")
    print(f"  ⚡  Rush Hour     : {'Yes' if rush else 'No'}")
    print(f"  🛣️  Road Condition : {road_label}")
    print(f"  🧠  Prediction    : {'ML Model' if using_ml else 'Rule-Based Fallback'}")
    print(f"  🔬  Inference Mode: {'Tiled (3×2)' if USE_TILING else 'Single-Pass'}")
    print("─" * 45)
    print(f"  🟢  Green Time    : {green_time:.1f} seconds")
    print("═" * 45)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 50)
    print("   🚦  AI Traffic Signal System  —  v2.0")
    print("═" * 50)

    image_path = input("\n📂 Image path (Enter = 'traffic.jpg'): ").strip()
    if not image_path:
        image_path = "traffic.jpg"

    print(f"\n[Setup] Loading YOLO ({MODEL_PATH})...")
    yolo  = YOLO(MODEL_PATH)
    brain = TrafficBrain()

    print("\n[Setup] Loading image...")
    frame = load_image(image_path)

    print("\n[Detection] Running detection pipeline...")
    cars, bikes, trucks, detections, roi_offset = perception_layer(yolo, frame)
    print(f"[Detection] 🚗 Cars: {cars}  🏍️ Bikes: {bikes}  🚌 Buses/Trucks: {trucks}")

    print("\n[Brain] Predicting green time...")
    green_time = brain.predict(cars, bikes, trucks)

    print_summary(cars, bikes, trucks, green_time, brain.using_ml)
    draw_detections(frame, cars, bikes, trucks, green_time, detections, roi_offset)
    simulate_signal_cycle(green_time)


if __name__ == "__main__":
    main()