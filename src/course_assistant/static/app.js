// Small progressive enhancements. Every page works without this file.
(function () {
  "use strict";
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.documentElement.classList.add("js");

  // Reader: scroll the cited passage into view and give it focus.
  function showTarget() {
    var id = (location.hash || "").slice(1);
    var el = id && document.getElementById(id);
    if (!el) { el = document.querySelector(".chunk.hit"); }
    if (el) {
      el.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" });
      el.focus({ preventScroll: true });
    }
  }
  window.addEventListener("load", showTarget);

  // Reader: seek a recording to a timestamp.
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("[data-seek]");
    if (!b) { return; }
    var p = document.getElementById("player");
    if (!p) { return; }
    p.currentTime = parseFloat(b.getAttribute("data-seek")) || 0;
    p.play && p.play();
    p.scrollIntoView({ block: "nearest" });
  });

  // Long AI calls: say what is happening, in a live region.
  var busy = document.getElementById("busy");
  document.body.addEventListener("htmx:beforeRequest", function (e) {
    var msg = e.detail.elt && e.detail.elt.getAttribute && e.detail.elt.getAttribute("data-busy");
    if (busy && msg) { busy.hidden = false; busy.textContent = msg; }
  });
  document.body.addEventListener("htmx:afterRequest", function () {
    if (busy) { busy.textContent = ""; busy.hidden = true; }
  });
  document.body.addEventListener("htmx:afterSwap", function () {
    var f = document.querySelector("#course-body .flash"); if (f) { f.setAttribute("tabindex", "-1"); }
  });
  // Plain form posts (no htmx): show the message and disable the button so it is not sent twice.
  document.addEventListener("submit", function (e) {
    var f = e.target, msg = f.getAttribute && f.getAttribute("data-busy");
    if (msg && !f.hasAttribute("hx-post") && busy) { busy.hidden = false; busy.textContent = msg; }
  });

  // Flashcards: hide the answer until revealed; space reveals, 1 to 4 grade.
  var deck = document.querySelector("[data-flashcards]");
  if (deck) {
    var reveal = document.getElementById("reveal"), ans = document.getElementById("answer");
    var show = function () { ans.hidden = false; reveal.hidden = true; var g = ans.querySelector("[data-key='3']"); g && g.focus(); };
    ans.hidden = true; reveal.hidden = false; reveal.addEventListener("click", show);
    document.addEventListener("keydown", function (e) {
      var t = e.target && e.target.tagName;
      if (e.ctrlKey || e.metaKey || e.altKey || t === "INPUT" || t === "TEXTAREA" || t === "SELECT") { return; }
      if (e.key === " " && ans.hidden) { e.preventDefault(); show(); return; }
      if (!ans.hidden && "1234".indexOf(e.key) !== -1 && e.key.length === 1) {
        var b = ans.querySelector("[data-key='" + e.key + "']"); if (b) { e.preventDefault(); b.click(); }
      }
    });
  }
})();
