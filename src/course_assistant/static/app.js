// Small progressive enhancements. Every page works without this file.
(function () {
  "use strict";
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.documentElement.classList.add("js");

  var status = document.getElementById("sr-status");
  var alertBox = document.getElementById("sr-alert");
  var toast = document.getElementById("busy");

  // Say something to a screen reader. The permanent live regions live outside #course-body, so they survive swaps.
  // The text is cleared first and set a moment later so the same sentence is announced again.
  function say(text, assertive) {
    var el = assertive ? alertBox : status;
    if (!el || !text) { return; }
    el.textContent = "";
    window.setTimeout(function () { el.textContent = text; }, 60);
  }
  function showToast(text) { if (toast && text) { toast.hidden = false; toast.textContent = text; } }
  function hideToast() { if (toast) { toast.textContent = ""; toast.hidden = true; } }
  function focusEl(el) {
    if (!el) { return false; }
    if (!el.hasAttribute("tabindex")) { el.setAttribute("tabindex", "-1"); }
    el.focus({ preventScroll: false });
    return true;
  }
  function submitButtons(form) { return form ? form.querySelectorAll("button[type='submit'], button:not([type])") : []; }
  function lock(form, on) {
    if (!form) { return; }
    if (on) { form.setAttribute("aria-busy", "true"); form.setAttribute("data-inflight", "1"); }
    else { form.removeAttribute("aria-busy"); form.removeAttribute("data-inflight"); }
    Array.prototype.forEach.call(submitButtons(form), function (b) {
      if (on) { if (!b.disabled) { b.setAttribute("data-was-enabled", "1"); b.disabled = true; } }
      else if (b.getAttribute("data-was-enabled")) { b.disabled = false; b.removeAttribute("data-was-enabled"); }
    });
  }
  function formOf(elt) { return elt && (elt.tagName === "FORM" ? elt : (elt.closest && elt.closest("form"))); }

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

  // A full page that follows a form post: say what happened and put focus on it.
  function announceFlash() {
    var invalid = document.querySelector("main [aria-invalid='true']");
    var flash = document.querySelector("main .flash.bad, main .flash.ok");
    if (flash) { say(flash.textContent.replace(/\s+/g, " ").trim(), flash.classList.contains("bad")); }
    if (invalid) { focusEl(invalid); } else if (flash) { focusEl(flash); }
  }
  window.addEventListener("load", function () {
    if (document.querySelector(".chunk.hit") || location.hash) { showTarget(); } else { announceFlash(); }
  });
  // Back/forward cache: a page restored after a long AI call must not stay locked.
  window.addEventListener("pageshow", function (e) {
    if (e.persisted) { hideToast(); Array.prototype.forEach.call(document.querySelectorAll("form[data-inflight]"), function (f) { lock(f, false); }); }
  });

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

  // ---- Long AI calls through htmx -------------------------------------------------------------------------
  var announced = false, hadBusy = false;
  function busyText(elt) {
    var holder = elt && elt.closest && elt.closest("[data-busy]");
    return holder ? holder.getAttribute("data-busy") : "";
  }
  document.body.addEventListener("htmx:beforeRequest", function (e) {
    var form = formOf(e.detail.elt);
    if (form && form.getAttribute("data-inflight")) { e.preventDefault(); return; } // never send two AI calls
    announced = false;
    var msg = busyText(e.detail.elt);
    hadBusy = !!msg;
    lock(form, true);
    var target = e.detail.target; if (target && target.setAttribute) { target.setAttribute("aria-busy", "true"); }
    if (msg) { showToast(msg); say(msg); }
  });
  // An error page the server built for this form (HTTP 4xx with the tab in it) is shown, not thrown away.
  document.body.addEventListener("htmx:beforeSwap", function (e) {
    var xhr = e.detail.xhr, code = xhr && xhr.status;
    if (code >= 400 && code < 500 && xhr.responseText && xhr.responseText.indexOf('id="course-body"') !== -1) {
      e.detail.shouldSwap = true;
      e.detail.isError = false;
    }
  });
  function failed(text) { announced = true; say(text, true); }
  document.body.addEventListener("htmx:responseError", function (e) {
    var xhr = e.detail.xhr;
    if (xhr && xhr.responseText && xhr.responseText.indexOf('id="course-body"') !== -1) { return; }
    failed("Something went wrong (error " + (xhr ? xhr.status : "unknown") + "). Nothing was changed. Try again.");
  });
  document.body.addEventListener("htmx:sendError", function () {
    failed("Could not reach Course Assistant. Check that the app is still running. Nothing was saved.");
  });
  document.body.addEventListener("htmx:timeout", function () {
    failed("This is taking too long and was stopped. Try again.");
  });
  document.body.addEventListener("htmx:afterRequest", function (e) {
    lock(formOf(e.detail.elt), false);
    var b = document.getElementById("course-body"); if (b) { b.removeAttribute("aria-busy"); }
    hideToast();
    if (hadBusy && !announced) { say("Done."); }
    hadBusy = false;
  });

  function requestPath(e) {
    var p = e.detail.pathInfo && e.detail.pathInfo.requestPath;
    if (p) { return String(p); }
    var el = e.detail.requestConfig && e.detail.requestConfig.elt || e.detail.elt;
    return (el && (el.getAttribute("hx-post") || el.getAttribute("hx-get"))) || "";
  }
  // After the course area is replaced: title, announcement, and focus on what is new.
  document.body.addEventListener("htmx:afterSettle", function (e) {
    var body = document.getElementById("course-body");
    if (!body || !body.hasAttribute("data-title")) { return; }
    document.title = body.getAttribute("data-title");
    var path = requestPath(e);
    var invalid = body.querySelector("[aria-invalid='true']");
    var bad = body.querySelector(".flash.bad");
    var ok = body.querySelector(".flash.ok");
    var message = (bad || ok) ? (bad || ok).textContent.replace(/\s+/g, " ").trim() : "";
    var tabSwitch = !!(e.detail.elt && e.detail.elt.hasAttribute && e.detail.elt.hasAttribute("data-tab-link"));
    var target = null;
    if (invalid) { target = invalid; }
    else if (bad) { target = bad; }
    else if (tabSwitch) { target = body.querySelector("#tab-panel h2"); }
    else if (/\/ask$/.test(path)) { var hs = body.querySelectorAll("[data-answer-heading]"); target = hs[hs.length - 1] || ok; }
    else if (/\/study\//.test(path)) { target = body.querySelector("#study-result h3") || body.querySelector("#study-result .flash") || ok; }
    else { target = ok || body.querySelector("#tab-panel h2"); }
    if (message) { announced = true; say((bad ? "" : "Done. ") + message, !!bad); }
    else if (tabSwitch) { announced = true; say(body.getAttribute("data-title").replace(/ - Course Assistant$/, "") + " page"); }
    focusEl(target);
  });

  // Plain form posts (no htmx), such as the dashboard primer buttons: say what is happening and lock the form so it
  // is not sent twice. The next page announces the result.
  document.addEventListener("submit", function (e) {
    var f = e.target;
    if (!f || !f.getAttribute) { return; }
    if (f.hasAttribute("hx-post")) { return; }
    if (f.getAttribute("data-inflight")) { e.preventDefault(); return; }
    var msg = f.getAttribute("data-busy");
    if (msg) { showToast(msg); say(msg); }
    window.setTimeout(function () { lock(f, true); }, 0);
  });

  // ---- Flashcards ------------------------------------------------------------------------------------------
  // Hide the answer until revealed. Space reveals and 1 to 4 grade, only while shortcuts are on (WCAG 2.1.4),
  // never inside a field, never with a modifier key.
  var deck = document.querySelector("[data-flashcards]");
  if (deck) {
    var reveal = document.getElementById("reveal"), ans = document.getElementById("answer");
    var back = document.getElementById("answer-text");
    var show = function () {
      ans.hidden = false; reveal.hidden = true;
      // Focus lands on the Back text, so it is read out; the grade buttons follow it in reading order.
      focusEl(back);
    };
    ans.hidden = true; reveal.hidden = false; reveal.addEventListener("click", show);
    var hints = deck.querySelectorAll(".key-hint");
    function applyHints() {
      var on = document.documentElement.getAttribute("data-shortcuts") !== "off";
      Array.prototype.forEach.call(hints, function (h) { h.hidden = !on; });
    }
    applyHints();
    document.addEventListener("keydown", function (e) {
      if (document.documentElement.getAttribute("data-shortcuts") === "off") { return; }
      if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) { return; }
      var t = e.target, tag = t && t.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (t && t.isContentEditable)) { return; }
      // Space on a focused link or button keeps its normal job.
      var interactive = tag === "A" || tag === "BUTTON" || tag === "SUMMARY";
      if (e.key === " " && ans.hidden && !interactive) { e.preventDefault(); show(); return; }
      if (!ans.hidden && e.key.length === 1 && "1234".indexOf(e.key) !== -1) {
        var b = ans.querySelector("[data-key='" + e.key + "']"); if (b) { e.preventDefault(); b.click(); }
      }
    });
  }
})();
