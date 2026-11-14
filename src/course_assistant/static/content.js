// Renders math in answers, guides and the reader with the bundled KaTeX (no network).
// Only elements the server marked as math are touched (.math, .math-block, [data-math]), so a price such as
// "$5 and $10" in ordinary text is never read as a formula. Output includes MathML so screen readers read the math.
(function () {
  "use strict";
  var SELECTOR = ".math, .math-block, [data-math]";
  var OPTIONS = {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false }
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    output: "htmlAndMathml",
    throwOnError: false,
    strict: false,
    trust: false
  };

  function renderAll(root) {
    if (typeof window.renderMathInElement !== "function") { return; }
    var scope = root && root.querySelectorAll ? root : document;
    var nodes = Array.prototype.slice.call(scope.querySelectorAll(SELECTOR));
    if (scope !== document && scope.matches && scope.matches(SELECTOR)) { nodes.unshift(scope); }
    nodes.forEach(function (el) {
      if (el.getAttribute("data-math-done")) { return; }
      if (el.closest("pre, code")) { return; }
      el.setAttribute("data-math-done", "1");
      try { window.renderMathInElement(el, OPTIONS); } catch (e) { /* leave the LaTeX text as it is */ }
    });
  }

  // Reader media. "Go to 0:49" moves the recording and plays only if it was already playing; "Pause" stops it and
  // says where, through the page's polite live region. This runs before app.js's own seek handler and replaces it
  // for these buttons so the spoken label matches the button's wording.
  function say(msg) {
    var el = document.getElementById("sr-status");
    if (!el) { return; }
    el.textContent = "";
    window.setTimeout(function () { el.textContent = msg; }, 30);
  }
  function clock(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return (h ? h + ":" + (m < 10 ? "0" : "") + m : m) + ":" + (s < 10 ? "0" : "") + s;
  }
  window.addEventListener("click", function (e) {
    var seek = e.target.closest && e.target.closest("[data-seek]");
    var p = document.getElementById("player");
    if (!seek || !p) { return; }
    e.stopPropagation();
    p.currentTime = parseFloat(seek.getAttribute("data-seek")) || 0;
    p.scrollIntoView({ block: "nearest" });
    var where = (seek.textContent || "").replace(/^\s*(?:Go to|Play from)\s*/, "").trim();
    say(p.paused ? "Recording is at " + where + ". Press play on the player to listen." : "Moved the recording to " + where + ". Still playing.");
  }, true);
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("[data-pause]");
    var p = document.getElementById("player");
    if (!b || !p) { return; }
    var was = p.paused;
    p.pause();
    say((was ? "Already paused at " : "Paused at ") + clock(p.currentTime) + ".");
  });

  window.addEventListener("load", function () { renderAll(document); });
  document.addEventListener("DOMContentLoaded", function () { renderAll(document); });
  document.addEventListener("htmx:afterSwap", function (e) { renderAll(e.target || document); });
  window.renderCourseMath = renderAll;
})();
