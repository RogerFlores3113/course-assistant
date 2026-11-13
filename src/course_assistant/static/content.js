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

  window.addEventListener("load", function () { renderAll(document); });
  document.addEventListener("DOMContentLoaded", function () { renderAll(document); });
  document.addEventListener("htmx:afterSwap", function (e) { renderAll(e.target || document); });
  window.renderCourseMath = renderAll;
})();
