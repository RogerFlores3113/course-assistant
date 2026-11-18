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
    if (el && el.querySelector) {  // "Add exam": land on the Title field, not on the card that holds it
      var title = el.querySelector("[data-focus-title]");
      if (title) { el = title; }
    }
    if (el) {
      el.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" });
      el.focus({ preventScroll: true });
      // A question box started for the student ("About X: "): the cursor goes after the text.
      if (el.hasAttribute && el.hasAttribute("data-caret-end") && el.setSelectionRange) {
        var n = el.value.length; el.setSelectionRange(n, n);
      }
    }
  }
  // "Jump to latest answer": move focus to the newest answer, not only the scroll position.
  document.addEventListener("click", function (e) {
    var a = e.target && e.target.closest && e.target.closest("[data-jump-latest]");
    if (!a) { return; }
    var t = document.getElementById("latest-answer");
    if (!t) { return; }
    e.preventDefault();
    var h = t.querySelector("[data-answer-heading]") || t;
    h.scrollIntoView({ block: "start", behavior: reduced ? "auto" : "smooth" });
    focusEl(h);
  });

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
    p.scrollIntoView({ block: "nearest" });
    // The button keeps focus. Play only if the person was already listening; otherwise say where it is parked.
    if (!p.paused) { say("Moved the recording to " + (b.textContent || "").replace(/^\s*Play from\s*/, "").trim() + ". Still playing."); }
    else { say("Recording is at " + (b.textContent || "").replace(/^\s*Play from\s*/, "").trim() + ". Press play on the player to listen."); }
  });

  // ---- Long AI calls through htmx -------------------------------------------------------------------------
  // htmx overwrites detail.elt with the swapped-in element before afterSettle, so the element the person
  // actually used is recorded here, when the request starts.
  var announced = false, clicked = null, clickedPath = "", pressed = null;
  function busyText(elt) {
    var holder = elt && elt.closest && elt.closest("[data-busy]");
    return holder ? holder.getAttribute("data-busy") : "";
  }
  function requestElt(e) { return (e.detail.requestConfig && e.detail.requestConfig.elt) || e.detail.elt; }
  function requestPath(e) {
    var p = e.detail.pathInfo && e.detail.pathInfo.requestPath;
    if (p) { return String(p); }
    var el = requestElt(e);
    return (el && el.getAttribute && (el.getAttribute("hx-post") || el.getAttribute("hx-get"))) || "";
  }
  // Background status polls (a photo or recording that is still being read) change one small block and must not
  // lock forms, show the busy toast, move focus or change the page title.
  var pollId = "";
  function isPoll(e) { var el = requestElt(e); return !!(el && el.hasAttribute && el.hasAttribute("data-poll")); }
  document.body.addEventListener("htmx:beforeRequest", function (e) {
    if (isPoll(e)) { pollId = requestElt(e).id || ""; return; }
    var elt = requestElt(e);
    var form = formOf(elt);
    if (form && form.getAttribute("data-inflight")) { e.preventDefault(); return; } // never send two AI calls
    announced = false;
    pressed = document.activeElement && document.activeElement.tagName === "BUTTON" ? document.activeElement : null;
    clicked = elt;
    clickedPath = requestPath(e);
    var msg = busyText(elt);
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
    if (isPoll(e)) { return; }
    var xhr = e.detail.xhr;
    if (xhr && xhr.responseText && xhr.responseText.indexOf('id="course-body"') !== -1) { return; }
    failed("Something went wrong (error " + (xhr ? xhr.status : "unknown") + "). Nothing was changed. Try again.");
  });
  document.body.addEventListener("htmx:sendError", function (e) {
    if (isPoll(e)) { return; }
    failed("Could not reach Course Assistant. Check that the app is still running. Nothing was saved.");
  });
  document.body.addEventListener("htmx:timeout", function (e) {
    if (isPoll(e)) { return; }
    failed("This is taking too long and was stopped. Try again.");
  });
  // Nothing is announced here. The result (or the error) is announced once, from afterSettle or from failed().
  document.body.addEventListener("htmx:afterRequest", function (e) {
    if (isPoll(e)) { return; }
    var form = formOf(requestElt(e));
    lock(form, false);
    // lock() disabled the button the person pressed. If nothing was swapped in (network error, timeout, 5xx), focus
    // would be lost with it, so put it back on that button (or the form's first submit button).
    var btn = pressed && document.body.contains(pressed) ? pressed : (form && document.body.contains(form) ? form.querySelector("button[type='submit'], button:not([type])") : null);
    var active = document.activeElement;
    if (btn && (!active || active === document.body || active === pressed || active.disabled)) { btn.disabled = false; btn.focus(); }
    pressed = null;
    var b = document.getElementById("course-body"); if (b) { b.removeAttribute("aria-busy"); }
    hideToast();
  });

  function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
  function leadingNumber(el) { var m = el && /^\s*(\d+)/.exec(el.textContent); return m ? parseInt(m[1], 10) : 0; }
  // "Answer ready. 3 sources, 1 citation removed, 1 quote not found."
  function answerSummary(scope, lead) {
    var parts = [plural(scope.querySelectorAll(".sources:not(.also) li").length, "source", "sources")];
    var removed = leadingNumber(scope.querySelector("[data-removed-citations]"));
    var quotes = leadingNumber(scope.querySelector("[data-unverified-quotes]"));
    if (removed) { parts.push(plural(removed, "citation", "citations") + " removed"); }
    if (quotes) { parts.push(plural(quotes, "quote", "quotes") + " not found"); }
    var partly = scope.querySelector("[data-partly-covered]") ? " Part of your question is not covered." : "";
    return lead + " " + parts.join(", ") + "." + partly;
  }
  // After the course area is replaced: title, one announcement, and focus on what is new.
  // One rule for GET tab links: announce "<Tab> page" and focus the tab heading. Everything else: focus the result.
  document.body.addEventListener("htmx:afterSettle", function (e) {
    if (pollId) {
      // A status poll finished: say so once, when the job is no longer waiting.
      var done = document.getElementById(pollId);
      pollId = "";
      if (done && !done.hasAttribute("data-poll")) {
        var line = done.querySelector(".status-note");
        say(line ? "Update: " + line.textContent.replace(/\s+/g, " ").trim() : "A recording or photo finished.");
      }
      return;
    }
    var body = document.getElementById("course-body");
    if (!body || !body.hasAttribute("data-title")) { return; }
    var title = body.getAttribute("data-title");
    document.title = title;
    var path = clickedPath || requestPath(e);
    var tabSwitch = !!(clicked && clicked.hasAttribute && clicked.hasAttribute("data-tab-link"));
    var invalid = body.querySelector("[aria-invalid='true']");
    var bad = body.querySelector(".flash.bad");
    var ok = body.querySelector(".flash.ok");
    var message = (bad || ok) ? (bad || ok).textContent.replace(/\s+/g, " ").trim() : "";
    var target = null, speech = "", loud = false;
    if (tabSwitch) {
      target = body.querySelector("#tab-panel h2");
      speech = title.split(" - ")[0] + " page";
    } else if (invalid || bad) {
      target = invalid || bad; speech = message; loud = true;
    } else if (/\/ask$/.test(path)) {
      var hs = body.querySelectorAll("[data-answer-heading]");
      target = hs[hs.length - 1] || ok;
      if (hs.length) { speech = answerSummary(hs[hs.length - 1].parentNode, "Answer ready."); }
      else if (message) { speech = message; }
    } else if (/\/study\//.test(path)) {
      var res = body.querySelector("#study-result");
      target = body.querySelector("#study-result h3") || body.querySelector("#study-result .flash") || ok;
      if (res && res.querySelector(".answer")) { speech = answerSummary(res, "Your guide is ready."); }
      else if (message) { speech = message; }
    } else {
      target = ok || body.querySelector("#tab-panel h2");
      speech = message;
    }
    if (speech && !announced) { say(speech, loud); }
    announced = false;
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

  // Flashcards end states (round finished, no cards): the page is a fresh load, so put focus on the heading and say it.
  window.addEventListener("load", function () {
    var end = document.querySelector("[data-end-state]");
    if (!end || location.hash || document.querySelector("main [aria-invalid='true'], main .flash")) { return; }
    var h = end.querySelector("h2");
    if (!h) { return; }
    var next = h.parentNode.querySelector("p");
    say((h.textContent + ". " + (next ? next.textContent : "")).replace(/\s+/g, " ").trim());
    focusEl(h);
  });

  // ---- Flashcards ------------------------------------------------------------------------------------------
  // Hide the answer until revealed. Space reveals and 1 to 4 grade, only while shortcuts are on (WCAG 2.1.4),
  // never inside a field, never with a modifier key.
  var deck = document.querySelector("[data-flashcards]");
  if (deck) {
    // After a grade the next card is a new page: put focus on its heading ("Card 2 of 4") unless a hash or flash won.
    window.addEventListener("load", function () {
      if (location.hash || document.querySelector(".chunk.hit") || document.querySelector("main [aria-invalid='true'], main .flash")) { return; }
      var h = deck.querySelector("h2"); if (h) { focusEl(h); }
    });
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
  // Copy buttons on code blocks (answers and guides). A real button after the block, reachable by keyboard,
  // that announces "Copied" through the polite live region. Without clipboard access it says so instead.
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) { return navigator.clipboard.writeText(text); }
    return new Promise(function (ok, no) {
      try {
        var t = document.createElement("textarea"); t.value = text; t.setAttribute("readonly", ""); t.style.position = "fixed"; t.style.opacity = "0";
        document.body.appendChild(t); t.select(); var done = document.execCommand("copy"); document.body.removeChild(t);
        if (done) { ok(); } else { no(); }
      } catch (err) { no(err); }
    });
  }
  // Every code block gets its own name ("julia code block 2 of 3") on the block and on its Copy button, so a screen
  // reader's list of buttons or regions can tell them apart. Numbers run over the answer the block sits in.
  function nameCodeBlocks(scope) {
    var pres = scope.querySelectorAll("pre[data-copy-ready]");
    Array.prototype.forEach.call(pres, function (pre, i) {
      var lang = pre.getAttribute("data-lang") || "";
      var what = (lang ? lang + " code" : "Code") + (pres.length > 1 ? " block " + (i + 1) + " of " + pres.length : "");
      pre.setAttribute("role", "group"); pre.setAttribute("aria-label", what);
      var b = pre.parentNode && pre.parentNode.querySelector(".copy-btn");
      if (b) { b.setAttribute("aria-label", "Copy " + what.charAt(0).toLowerCase() + what.slice(1)); }
    });
  }
  function addCopyButtons(root) {
    var pres = (root || document).querySelectorAll("pre");
    Array.prototype.forEach.call(pres, function (pre) {
      if (pre.hasAttribute("data-copy-ready") || pre.closest(".copy-wrap")) { return; }
      pre.setAttribute("data-copy-ready", "1");
      var wrap = document.createElement("div"); wrap.className = "copy-wrap";
      pre.parentNode.insertBefore(wrap, pre); wrap.appendChild(pre);
      var b = document.createElement("button"); b.type = "button"; b.className = "btn small ghost copy-btn";
      b.textContent = "Copy"; b.setAttribute("aria-label", "Copy code");
      b.addEventListener("click", function () {
        var code = pre.querySelector("code") || pre;
        copyText(code.textContent).then(function () {
          b.textContent = "Copied"; say("Copied");
          window.setTimeout(function () { b.textContent = "Copy"; }, 2000);
        }, function () { say("Could not copy. Select the code and copy it by hand."); });
      });
      wrap.appendChild(b);
    });
    var scopes = [];
    Array.prototype.forEach.call((root || document).querySelectorAll("pre[data-copy-ready]"), function (pre) {
      var sc = pre.closest("[data-latest-answer], .msg, article, main") || document.body;
      if (scopes.indexOf(sc) < 0) { scopes.push(sc); }
    });
    scopes.forEach(nameCodeBlocks);
  }
  addCopyButtons(document);
  document.body.addEventListener("htmx:afterSwap", function (e) { addCopyButtons(e.target || document); });
})();
