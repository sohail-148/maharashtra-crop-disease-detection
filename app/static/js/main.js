/**
 * main.js — Global UI behaviour
 * - Mobile nav toggle
 * - Flash message dismissal
 * - Tab switching (Upload / Camera)
 */

(function () {
  "use strict";

  /* ----------------------------------------------------------------
     Mobile nav toggle
     ---------------------------------------------------------------- */
  const navToggle = document.getElementById("navToggle");
  const navLinks  = document.getElementById("navLinks");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      const isOpen = navLinks.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
    });

    // Close on outside click
    document.addEventListener("click", function (e) {
      if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
        navLinks.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      }
    });

    // Close on Escape
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && navLinks.classList.contains("is-open")) {
        navLinks.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
        navToggle.focus();
      }
    });
  }

  /* ----------------------------------------------------------------
     Flash message dismissal
     ---------------------------------------------------------------- */
  document.querySelectorAll(".flash-close").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const flash = btn.closest(".flash");
      if (flash) {
        flash.style.opacity = "0";
        flash.style.transition = "opacity 200ms";
        setTimeout(function () { flash.remove(); }, 210);
      }
    });
  });

  /* ----------------------------------------------------------------
     Tab switching — Upload / Camera
     ---------------------------------------------------------------- */
  const tabs = document.querySelectorAll(".tab[role='tab']");

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      const panelId = tab.getAttribute("aria-controls");

      // Deactivate all tabs and hide all panels
      tabs.forEach(function (t) {
        t.classList.remove("tab--active");
        t.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach(function (p) {
        p.hidden = true;
      });

      // Activate clicked tab
      tab.classList.add("tab--active");
      tab.setAttribute("aria-selected", "true");
      const panel = document.getElementById(panelId);
      if (panel) panel.hidden = false;
    });
  });

})();
