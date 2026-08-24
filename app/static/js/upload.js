/**
 * upload.js — Drag-and-drop / click file upload with image preview
 *
 * Responsibilities:
 *  - Open file picker on dropzone click or Enter/Space keypress
 *  - Accept drag-and-drop events
 *  - Validate file type and size (client-side, mirrors server limits)
 *  - Show inline preview; allow removal
 *  - Clear camera data when a file is chosen (only one input method active)
 */

(function () {
  "use strict";

  const dropzone       = document.getElementById("dropzone");
  const fileInput      = document.getElementById("fileInput");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const dropzonePreview= document.getElementById("dropzonePreview");
  const previewImg     = document.getElementById("previewImg");
  const previewFilename= document.getElementById("previewFilename");
  const removeBtn      = document.getElementById("removeImage");
  const cameraData     = document.getElementById("cameraImageData");
  const analyseBtn     = document.getElementById("analyseBtn");

  if (!dropzone || !fileInput) return;  // not on upload page

  const MAX_BYTES    = 10 * 1024 * 1024;  // 10 MB
  const ALLOWED_TYPES= ["image/jpeg", "image/png", "image/bmp", "image/webp"];
  const ALLOWED_EXT  = /\.(jpe?g|png|bmp|webp)$/i;

  /* ----------------------------------------------------------------
     Show / hide preview
     ---------------------------------------------------------------- */
  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewFilename.textContent = file.name + " (" +
        (file.size / 1024).toFixed(0) + " KB)";
      dropzonePrompt.hidden  = true;
      dropzonePreview.hidden = false;
      // clear camera data so only file path is submitted
      if (cameraData) cameraData.value = "";
    };
    reader.readAsDataURL(file);
  }

  function clearPreview() {
    previewImg.src = "";
    previewFilename.textContent = "";
    dropzonePrompt.hidden  = false;
    dropzonePreview.hidden = true;
    fileInput.value = "";   // reset input so same file can be re-selected
  }

  /* ----------------------------------------------------------------
     Validation
     ---------------------------------------------------------------- */
  function validateFile(file) {
    if (!file) return "No file selected.";
    if (!ALLOWED_TYPES.includes(file.type) && !ALLOWED_EXT.test(file.name)) {
      return "Unsupported file type. Please upload a JPG, PNG, BMP, or WEBP image.";
    }
    if (file.size > MAX_BYTES) {
      return "File is too large (max 10 MB).";
    }
    return null;  // valid
  }

  function showError(msg) {
    // Reuse flash mechanism: insert a flash div at the top of the page
    const container = document.querySelector(".flash-container") ||
      (() => {
        const c = document.createElement("div");
        c.className = "flash-container";
        document.querySelector("main").prepend(c);
        return c;
      })();

    const div = document.createElement("div");
    div.className = "flash flash--error";
    div.setAttribute("role", "alert");
    div.innerHTML = `<span>${msg}</span>
      <button class="flash-close" aria-label="Dismiss">&times;</button>`;
    div.querySelector(".flash-close").addEventListener("click", () => div.remove());
    container.prepend(div);
    setTimeout(() => div.remove(), 6000);
  }

  /* ----------------------------------------------------------------
     File input change
     ---------------------------------------------------------------- */
  fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];
    if (!file) return;
    const err = validateFile(file);
    if (err) { showError(err); fileInput.value = ""; return; }
    showPreview(file);
  });

  /* ----------------------------------------------------------------
     Click / keyboard on dropzone opens file picker
     ---------------------------------------------------------------- */
  dropzone.addEventListener("click", function (e) {
    // Prevent double-trigger when clicking the actual input or remove button
    if (e.target === fileInput || e.target === removeBtn) return;
    if (dropzonePreview && !dropzonePreview.hidden) return; // already has file
    fileInput.click();
  });

  dropzone.addEventListener("keydown", function (e) {
    if ((e.key === "Enter" || e.key === " ") && dropzonePreview.hidden) {
      e.preventDefault();
      fileInput.click();
    }
  });

  /* ----------------------------------------------------------------
     Drag and drop
     ---------------------------------------------------------------- */
  ["dragenter", "dragover"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend", "drop"].forEach(function (evt) {
    dropzone.addEventListener(evt, function () {
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const err = validateFile(file);
    if (err) { showError(err); return; }

    // Inject the dropped file into the input
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;

    showPreview(file);
  });

  /* ----------------------------------------------------------------
     Remove button
     ---------------------------------------------------------------- */
  if (removeBtn) {
    removeBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      clearPreview();
    });
  }

  /* ----------------------------------------------------------------
     Form submit guard — ensure at least one image source is active
     ---------------------------------------------------------------- */
  const form = document.getElementById("analyseForm");
  if (form) {
    form.addEventListener("submit", function (e) {
      const hasFile   = fileInput.files && fileInput.files.length > 0;
      const hasCamera = cameraData && cameraData.value.length > 0;
      if (!hasFile && !hasCamera) {
        e.preventDefault();
        showError("Please upload or capture a leaf image before analysing.");
      }
    });
  }

})();
