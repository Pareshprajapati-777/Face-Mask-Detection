import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera as CameraIcon, CameraOff, Video, AlertCircle } from 'lucide-react';
import { SmoothDetectionTracker } from './DetectionBox';

const Camera = ({ onDetectionResults, onErrorChange, isAiOnline }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const animFrameRef = useRef(null);
  const trackerRef = useRef(new SmoothDetectionTracker());

  const isRequestPendingRef = useRef(false);
  const lastFrameTimeRef = useRef(Date.now());

  // 60 FPS continuous animation loop for smooth bounding box gliding
  const startAnimationLoop = useCallback(() => {
    const animate = () => {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        trackerRef.current.stepAndRender(ctx);
      }
      animFrameRef.current = requestAnimationFrame(animate);
    };

    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
    animFrameRef.current = requestAnimationFrame(animate);
  }, []);

  // Clear canvas overlay
  const clearCanvas = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    trackerRef.current.reset();
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, []);

  // Stop camera feed and cleanup
  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
    isRequestPendingRef.current = false;
    clearCanvas();
  }, [clearCanvas]);

  // Capture frame & send to Node Express Gateway (/api/detect)
  const captureAndSendFrame = useCallback(async () => {
    if (
      !videoRef.current ||
      !canvasRef.current ||
      videoRef.current.readyState !== 4 ||
      isRequestPendingRef.current
    ) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Set canvas dimensions to match video dimensions
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Lock to prevent concurrent overlapping requests
    isRequestPendingRef.current = true;

    try {
      // Create off-screen canvas to extract JPEG blob
      const offscreenCanvas = document.createElement('canvas');
      offscreenCanvas.width = Math.min(640, video.videoWidth);
      offscreenCanvas.height = Math.min(480, video.videoHeight);
      
      const offscreenCtx = offscreenCanvas.getContext('2d');
      offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

      offscreenCanvas.toBlob(
        async (blob) => {
          if (!blob) {
            isRequestPendingRef.current = false;
            return;
          }

          const formData = new FormData();
          formData.append('frame', blob, 'frame.jpg');

          try {
            const response = await fetch('/api/detect', {
              method: 'POST',
              body: formData
            });

            if (!response.ok) {
              const errData = await response.json().catch(() => ({}));
              throw new Error(errData.message || `Server Error: ${response.status}`);
            }

            const data = await response.json();

            // Calculate FPS
            const now = Date.now();
            const deltaSec = (now - lastFrameTimeRef.current) / 1000;
            lastFrameTimeRef.current = now;
            const currentFps = deltaSec > 0 ? Math.round(1 / deltaSec) : 0;

            // Pass targets to 60 FPS smooth lerp tracker
            const scaleX = video.videoWidth / offscreenCanvas.width;
            const scaleY = video.videoHeight / offscreenCanvas.height;
            trackerRef.current.updateTargets(data.faces, scaleX, scaleY);

            if (onDetectionResults) {
              onDetectionResults(data, currentFps);
            }
            if (onErrorChange) onErrorChange(null);
          } catch (err) {
            console.error('[DETECTION ERROR]', err.message);
            if (onErrorChange) {
              onErrorChange(err.message.includes('offline') ? 'AI Server is currently offline.' : err.message);
            }
          } finally {
            isRequestPendingRef.current = false;
          }
        },
        'image/jpeg',
        0.75
      );
    } catch (err) {
      console.error('[FRAME CAPTURE ERROR]', err);
      isRequestPendingRef.current = false;
    }
  }, [onDetectionResults, onErrorChange]);

  // Start webcam feed
  const startCamera = async () => {
    setErrorMsg(null);
    if (onErrorChange) onErrorChange(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      setIsStreaming(true);
      startAnimationLoop();

      // Start throttled frame capture loop (every 180ms ~ 5.5 requests/sec max)
      intervalRef.current = setInterval(() => {
        captureAndSendFrame();
      }, 180);

    } catch (err) {
      console.error('[CAMERA ACCESS ERROR]', err);
      let msg = 'Failed to access camera.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Camera permission denied. Please allow camera access in your browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No camera device found on this system.';
      }
      setErrorMsg(msg);
      if (onErrorChange) onErrorChange(msg);
      setIsStreaming(false);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div className="camera-card">
      <div className="camera-wrapper">
        <video
          ref={videoRef}
          className="video-feed"
          playsInline
          muted
          style={{ display: isStreaming ? 'block' : 'none' }}
        />
        <canvas ref={canvasRef} className="overlay-canvas" />

        {!isStreaming && (
          <div className="camera-placeholder">
            <Video size={48} opacity={0.4} />
            <p>Webcam is currently turned off.</p>
            {errorMsg && (
              <div className="alert-banner error" style={{ marginTop: '12px' }}>
                <AlertCircle size={18} />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="camera-controls">
        {!isStreaming ? (
          <button className="btn btn-primary" onClick={startCamera}>
            <CameraIcon size={18} />
            <span>START CAMERA</span>
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopCamera}>
            <CameraOff size={18} />
            <span>STOP CAMERA</span>
          </button>
        )}
      </div>
    </div>
  );
};

export default Camera;
