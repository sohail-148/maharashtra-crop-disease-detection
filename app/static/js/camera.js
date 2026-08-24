/**
 * camera.js — Browser camera capture
 *
 * Responsibilities:
 *  - Request camera permission via getUserMedia
 *  - Stream live feed into <video>
 *  - Capture a frame onto <canvas>, convert to JPEG blob
 *  - Store base64 data in hidden input for form submission
 *  - Allow retake; clear file input when camera is used
 *  - Graceful fallback with user-friendly message if camera unavailable
 */

(function () {
  "use strict";

  const startBtn       = document.getElementById("startCameraBtn");
  const captureBtn     = document.getElementById("captureBtn");
  const retakeBtn      = document.getElementById("retakeBtn");
  const videoEl        = document.getElementById("cameraFeed");
  const canvasEl       = document.getElementById("cameraCanvas");
  const capturePreview = document.getElementById("capturePreview");
  const capturedImg    = document.getElementById("capturedImg");
  const cameraData     = document.getElementById("cameraImageData");
  const cameraHint     = document.getElementById("cameraHint");
  const fileInput      = document.getElementById("fileInput");

  if (!startBtn || !videoEl) return;   // not on camera-capable page

  let stream = null;

  /* ----------------------------------------------------------------
     Helpers
     ---------------------------------------------------------------- */
  function setHint(msg, isError) {
    if (!cameraHint) return;
    cameraHint.textContent = msg;
    cameraHint.style.color = isError ? "var(--color-danger)" : "var(--color-text-muted)";
  }

  function stopStream() {
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    videoEl.srcObject = null;
  }

  /* ----------------------------------------------------------------
     Start camera
     ---------------------------------------------------------------- */
  startBtn.addEventListener("click", async function () {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setHint("Camera not supported in this browser.", true);
      return;
    }

    try {
      setHint("Requesting camera permission…");
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();

      startBtn.disabled           = true;
      startBtn.setAttribute("aria-disabled", "true");
      captureBtn.disabled         = false;
      captureBtn.removeAttribute("aria-disabled");
      setHint("Camera active. Hold the leaf steady, then press Capture.");
    } catch (err) {
      const msgs = {
        NotAllowedError:  "Camera permission denied. Please allow access in your browser settings.",
        NotFoundError:    "No camera found on this device.",
        NotReadableError: "Camera is in use by another application.",
        OverconstrainedError: "Camera does not meet the required constraints.",
      };
      setHint(msgs[err.name] || ("Camera error: " + err.message), true);
    }
  });

  /* ----------------------------------------------------------------
     Capture frame
     ---------------------------------------------------------------- */
  captureBtn.addEventListener("click", function () {
    if (!stream) return;

    // Size canvas to match video feed
    canvasEl.width  = videoEl.videoWidth  || 640;
    canvasEl.height = videoEl.videoHeight || 480;

    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

    // Convert to base64 JPEG (quality 0.92)
    const dataUrl = canvasEl.toDataURL("image/jpeg", 0.92);

    // Store in hidden input for form submission
    cameraData.value = dataUrl;

    // Show preview
    capturedImg.src      = dataUrl;
    capturePreview.hidden = false;
    videoEl.hidden        = true;

    // Stop stream to turn off camera indicator LED
    stopStream();

    captureBtn.disabled = true;
    captureBtn.setAttribute("aria-disabled", "true");
    retakeBtn.hidden = false;

    // Clear any file upload so only camera data is submitted
    if (fileInput) {
      fileInput.value = "";
      const dropzonePreview = document.getElementById("dropzonePreview");
      const dropzonePrompt  = document.getElementById("dropzonePrompt");
      if (dropzonePreview) dropzonePreview.hidden = true;
      if (dropzonePrompt)  dropzonePrompt.hidden  = false;
    }

    setHint("Image captured. Press Retake to try again, or Analyse Leaf to submit.");
  });

  /* ----------------------------------------------------------------
     Retake
     ---------------------------------------------------------------- */
  retakeBtn.addEventListener("click", async function () {
    // Clear captured data
    cameraData.value      = "";
    capturedImg.src       = "";
    capturePreview.hidden = true;
    videoEl.hidden        = false;
    retakeBtn.hidden      = true;

    // Restart stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
      captureBtn.disabled = false;
      captureBtn.removeAttribute("aria-disabled");
      setHint("Camera active. Hold the leaf steady, then press Capture.");
    } catch (err) {
      setHint("Could not restart camera: " + err.message, true);
    }
  });

  /* ----------------------------------------------------------------
     Stop camera when the Upload tab is selected
     ---------------------------------------------------------------- */
  const uploadTab = document.getElementById("tab-upload");
  if (uploadTab) {
    uploadTab.addEventListener("click", function () {
      stopStream();
      cameraData.value = "";
      capturePreview.hidden = true;
      videoEl.hidden        = false;
      startBtn.disabled     = false;
      startBtn.removeAttribute("aria-disabled");
      captureBtn.disabled   = true;
      captureBtn.setAttribute("aria-disabled", "true");
      setHint("Camera requires browser permission. Works in Chrome, Firefox, Safari.");
    });
  }

  /* ----------------------------------------------------------------
     Clean up on page unload
     ---------------------------------------------------------------- */
  window.addEventListener("beforeunload", stopStream);

})();
