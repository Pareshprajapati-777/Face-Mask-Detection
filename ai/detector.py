import cv2
import numpy as np
from PIL import Image
from ai.config import CONFIDENCE_THRESHOLD, IMG_HEIGHT, IMG_WIDTH
from ai.model_loader import ModelLoader
from ai.face_detector import FaceDetector

def is_valid_face_region(gray_roi, w, h, frame_w, frame_h):
    """
    Filter out false positive detection regions such as light fixtures, background doors, and reflections.
    """
    if w <= 0 or h <= 0:
        return False

    # 1. Aspect Ratio Filter (Human faces usually have ratio between 0.62 and 1.30)
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.62 or aspect_ratio > 1.30:
        return False

    # 2. Minimum/Maximum Relative Size Filter
    roi_area = w * h
    frame_area = frame_w * frame_h
    if roi_area < 900 or (frame_area > 0 and roi_area / frame_area > 0.85):
        return False

    # 3. Bright Light / Overexposure / Reflection Filter
    # Light fixtures produce extremely high mean brightness (>218) with low variance or blown-out white areas
    mean_val = np.mean(gray_roi)
    std_val = np.std(gray_roi)

    if mean_val > 225:
        return False  # Overexposed light bulb / fixture

    if mean_val > 210 and std_val < 30:
        return False  # Uniform bright light glow / reflection

    if std_val < 10:
        return False  # Completely flat/featureless region

    return True


def apply_nms(boxes, overlap_thresh=0.4):
    """
    Non-Maximum Suppression (NMS) to remove duplicate overlapping bounding boxes.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    w = boxes_arr[:, 2]
    h = boxes_arr[:, 3]
    x2 = x1 + w
    y2 = y1 + h

    areas = w * h
    order = np.argsort(areas)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= overlap_thresh)[0]
        order = order[inds + 1]

    return [boxes[k] for k in keep]


def process_frame_bytes(image_bytes: bytes):
    """
    Decode image frame, detect visible faces, apply strict false-positive filtering,
    run CNN inference, and return prediction results.
    """
    # 1. Decode bytes into OpenCV BGR numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Invalid or unreadable image frame bytes.")

    frame_h, frame_w = frame.shape[:2]

    # 2. Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect all candidate face regions
    face_detector = FaceDetector()
    raw_faces = face_detector.detect_faces(gray)

    if len(raw_faces) == 0:
        return {
            "faces": [],
            "total_faces": 0,
            "mask_count": 0,
            "no_mask_count": 0,
            "unknown_count": 0
        }

    # 4. Filter out non-face regions (lights, doors, background noise)
    valid_boxes = []
    for (x, y, w, h) in raw_faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
        bw, bh = x2 - x1, y2 - y1

        gray_roi = gray[y1:y2, x1:x2]
        if gray_roi.size == 0:
            continue

        if is_valid_face_region(gray_roi, bw, bh, frame_w, frame_h):
            valid_boxes.append((x1, y1, bw, bh))

    # 5. Apply Non-Maximum Suppression to prevent duplicate boxes
    final_boxes = apply_nms(valid_boxes, overlap_thresh=0.35)

    model_loader = ModelLoader()
    model = model_loader.model
    class_names = model_loader.class_names

    detected_faces = []
    mask_count = 0
    no_mask_count = 0
    unknown_count = 0

    # 6. Loop over verified face ROIs and run CNN inference
    for (x1, y1, bw, bh) in final_boxes:
        x2, y2 = x1 + bw, y1 + bh
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0 or face_roi.shape[0] < 12 or face_roi.shape[1] < 12:
            continue

        # Preprocessing: BGR -> RGB, Resize 224x224, Normalize [0, 1]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_WIDTH, IMG_HEIGHT))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)

        # CNN Softmax Inference
        raw_probs = model.predict(face_batch, verbose=0)[0]
        max_idx = int(np.argmax(raw_probs))
        max_conf = float(raw_probs[max_idx])

        # Get label from saved class_names mapping
        predicted_label = class_names.get(str(max_idx), "UNKNOWN")

        # Apply Confidence Threshold
        if max_conf < CONFIDENCE_THRESHOLD:
            final_label = "UNKNOWN"
            unknown_count += 1
        elif predicted_label == "MASK":
            final_label = "MASK"
            mask_count += 1
        elif predicted_label == "NO_MASK":
            final_label = "NO_MASK"
            no_mask_count += 1
        else:
            final_label = "UNKNOWN"
            unknown_count += 1

        detected_faces.append({
            "x": int(x1),
            "y": int(y1),
            "width": int(bw),
            "height": int(bh),
            "label": final_label,
            "confidence": round(max_conf, 4)
        })

    return {
        "faces": detected_faces,
        "total_faces": len(detected_faces),
        "mask_count": mask_count,
        "no_mask_count": no_mask_count,
        "unknown_count": unknown_count
    }

