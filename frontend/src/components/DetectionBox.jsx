/**
 * Smooth Bounding Box Interpolator & Canvas Renderer
 * Converts choppy discrete API detection coordinates into silky smooth 60 FPS gliding animations.
 */
export class SmoothDetectionTracker {
  constructor() {
    this.trackedFaces = [];
    this.nextId = 1;
  }

  reset() {
    this.trackedFaces = [];
  }

  /**
   * Calculate Intersection over Union (IoU) between two bounding boxes
   */
  calculateIoU(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
    const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (interArea === 0) return 0;

    const boxAArea = boxA.width * boxA.height;
    const boxBArea = boxB.width * boxB.height;
    const iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
  }

  /**
   * Update target coordinates when a new frame result comes from API
   */
  updateTargets(faces, scaleX = 1, scaleY = 1) {
    const now = Date.now();

    if (!faces || faces.length === 0) {
      // Mark all current faces as unassigned for fadeout hysteresis
      this.trackedFaces.forEach((tf) => {
        tf.updatedThisFrame = false;
      });
      return;
    }

    const scaledFaces = faces.map((f) => ({
      x: f.x * scaleX,
      y: f.y * scaleY,
      width: f.width * scaleX,
      height: f.height * scaleY,
      label: f.label,
      confidence: f.confidence
    }));

    const assignedTracked = new Set();
    const assignedNew = new Set();

    // 1. Match new detections to existing tracked faces using IoU
    for (let i = 0; i < scaledFaces.length; i++) {
      const newFace = scaledFaces[i];
      let bestIoU = 0.20; // Minimum IoU threshold to consider match
      let bestMatchIdx = -1;

      for (let j = 0; j < this.trackedFaces.length; j++) {
        if (assignedTracked.has(j)) continue;

        const tracked = this.trackedFaces[j];
        const targetBox = {
          x: tracked.targetX,
          y: tracked.targetY,
          width: tracked.targetW,
          height: tracked.targetH
        };

        const iou = this.calculateIoU(newFace, targetBox);
        if (iou > bestIoU) {
          bestIoU = iou;
          bestMatchIdx = j;
        }
      }

      if (bestMatchIdx !== -1) {
        assignedTracked.add(bestMatchIdx);
        assignedNew.add(i);

        const match = this.trackedFaces[bestMatchIdx];
        match.targetX = newFace.x;
        match.targetY = newFace.y;
        match.targetW = newFace.width;
        match.targetH = newFace.height;
        match.label = newFace.label;
        match.confidence = newFace.confidence;
        match.lastSeen = now;
        match.updatedThisFrame = true;
      }
    }

    // 2. Create new tracked items for unmatched detections
    for (let i = 0; i < scaledFaces.length; i++) {
      if (assignedNew.has(i)) continue;

      const newFace = scaledFaces[i];
      this.trackedFaces.push({
        id: this.nextId++,
        currX: newFace.x,
        currY: newFace.y,
        currW: newFace.width,
        currH: newFace.height,
        targetX: newFace.x,
        targetY: newFace.y,
        targetW: newFace.width,
        targetH: newFace.height,
        label: newFace.label,
        confidence: newFace.confidence,
        lastSeen: now,
        opacity: 0,
        targetOpacity: 1,
        updatedThisFrame: true
      });
    }

    // Mark unassigned existing faces for fadeout
    for (let j = 0; j < this.trackedFaces.length; j++) {
      if (!assignedTracked.has(j) && !assignedNew.has(j)) {
        this.trackedFaces[j].updatedThisFrame = false;
      }
    }
  }

  /**
   * Step interpolation forward and draw clean boxes on 60 FPS animation frame
   */
  stepAndRender(ctx) {
    if (!ctx || !ctx.canvas) return;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const now = Date.now();
    const lerpFactor = 0.22; // Smooth linear interpolation speed (0.15 = buttery smooth, 0.30 = fast)
    const fadeOutDelayMs = 350; // Keep box visible for 350ms during single-frame drops

    // Filter out expired faces
    this.trackedFaces = this.trackedFaces.filter((face) => {
      const timeSinceSeen = now - face.lastSeen;
      if (!face.updatedThisFrame && timeSinceSeen > fadeOutDelayMs) {
        face.targetOpacity = 0;
      } else {
        face.targetOpacity = 1;
      }

      // Smooth opacity fade
      face.opacity += (face.targetOpacity - face.opacity) * 0.2;

      return face.opacity > 0.02;
    });

    // Interpolate positions and render each tracked face
    this.trackedFaces.forEach((face) => {
      face.currX += (face.targetX - face.currX) * lerpFactor;
      face.currY += (face.targetY - face.currY) * lerpFactor;
      face.currW += (face.targetW - face.currW) * lerpFactor;
      face.currH += (face.targetH - face.currH) * lerpFactor;

      const x = face.currX;
      const y = face.currY;
      const width = face.currW;
      const height = face.currH;
      const confidencePct = (face.confidence * 100).toFixed(2);

      let color = '#f59e0b'; // Yellow for UNKNOWN
      let textLabel = `UNKNOWN ${confidencePct}%`;

      if (face.label === 'MASK') {
        color = '#22c55e'; // Green for MASK
        textLabel = `MASK ${confidencePct}%`;
      } else if (face.label === 'NO_MASK') {
        color = '#ef4444'; // Red for NO_MASK
        textLabel = `NO MASK ${confidencePct}%`;
      }

      ctx.save();
      ctx.globalAlpha = Math.max(0, Math.min(1, face.opacity));

      // 1. Draw Bounding Box Rectangle with Smooth Glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;

      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 8);
        ctx.stroke();
      } else {
        ctx.strokeRect(x, y, width, height);
      }

      // Reset shadow for label rendering
      ctx.shadowBlur = 0;

      // 2. Draw Label Header Background Box
      ctx.font = 'bold 13px Inter, sans-serif';
      const textMetrics = ctx.measureText(textLabel);
      const textWidth = textMetrics.width;
      const headerHeight = 26;
      const headerY = Math.max(0, y - headerHeight);

      ctx.fillStyle = color;
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, headerY, textWidth + 14, headerHeight, [6, 6, 0, 0]);
        ctx.fill();
      } else {
        ctx.fillRect(x, headerY, textWidth + 14, headerHeight);
      }

      // 3. Draw Text Label
      ctx.fillStyle = '#ffffff';
      ctx.fillText(textLabel, x + 7, headerY + 18);

      ctx.restore();
    });
  }
}

/**
 * Fallback static function for simple one-off draws
 */
export const drawBoundingBoxes = (ctx, faces, scaleX, scaleY) => {
  if (!window._globalTracker) {
    window._globalTracker = new SmoothDetectionTracker();
  }
  window._globalTracker.updateTargets(faces, scaleX, scaleY);
  window._globalTracker.stepAndRender(ctx);
};

