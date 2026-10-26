"""Shared UC3.2-style HTML chrome for the SoMES form and dashboard."""

from __future__ import annotations

import html
import json
from typing import Any

# Visual tokens, layout, tooltips, bill/mrk/hw strips and day picker — same as
# UC3.2 SEED `DashboardPiece/render.py`. Extra rules at the end are form-only.
CSS = """
:root {
  --bg: #f5f6f8;
  --panel: #ffffff;
  --ink: #14181f;
  --muted: #5c6675;
  --line: #dfe3e9;
  --accent: #1a6a8c;
  --good: #2f8f5b;
  --warn: #b4761e;
  --bad: #b23a3a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
header { border-bottom: 2px solid var(--ink); padding-bottom: 16px; margin-bottom: 14px; }
.meta-strip {
  display: flex; align-items: flex-start; gap: 10px;
  margin: 0 0 22px; padding: 7px 12px;
  background: #e8eef2; border: 1px solid #d3dce4; border-radius: 8px;
  font-size: 13px; line-height: 1.45; color: var(--muted);
}
.meta-strip .meta-k {
  flex: 0 0 auto; font-size: 11px; font-weight: 650;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent);
  padding-top: 1px; white-space: nowrap;
}
.meta-strip p { margin: 0; }
.kicker {
  font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent); font-weight: 650; margin: 0 0 8px;
}
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 14px; }
.source-box { border-left: 3px solid var(--accent); padding: 2px 0 2px 16px; margin-bottom: 16px; }
.source-box strong { display: block; font-size: 16px; margin-bottom: 4px; }
h2 {
  font-size: 15px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 36px 0 14px; font-weight: 600;
}
.tip {
  display: inline-flex; align-items: center; justify-content: center;
  width: 15px; height: 15px; margin-left: 6px; border-radius: 6px;
  border: 1px solid currentColor; font-size: 10px; font-weight: 700;
  line-height: 1; cursor: help; position: relative; vertical-align: 1px;
  flex: 0 0 auto; text-transform: none; letter-spacing: 0; opacity: .72;
  border-radius: 50%;
}
.tip:hover, .tip:focus { opacity: 1; outline: none; z-index: 5; }
.tip .tip-text,
.card .label .tip .tip-text,
h2 .tip .tip-text,
th .tip .tip-text,
td .tip .tip-text,
label .tip .tip-text {
  visibility: hidden; opacity: 0; pointer-events: none;
  position: absolute; z-index: 80; top: calc(100% + 8px); left: 0;
  bottom: auto; transform: none;
  width: max-content; max-width: min(360px, 70vw);
  padding: 10px 12px; border-radius: 6px; background: #1b212b; color: #f4f6f8;
  font-size: 12px; font-weight: 400; line-height: 1.45; text-transform: none;
  letter-spacing: 0; text-align: left; box-shadow: 0 8px 22px rgba(20,24,31,.22);
  white-space: normal;
}
.card .label .tip .tip-text { left: auto; right: 0; }
.tip:hover .tip-text, .tip:focus .tip-text,
.card .label .tip:hover .tip-text, .card .label .tip:focus .tip-text {
  visibility: visible; opacity: 1;
}
.panel {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 18px; overflow: visible;
}
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; overflow: visible; }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; overflow: visible; position: relative; z-index: 1;
}
.card:hover, .card:focus-within { z-index: 30; }
.card .label {
  display: flex; align-items: flex-start; gap: 2px;
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); line-height: 1.35;
}
.card .label > span:not(.tip) { flex: 1; min-width: 0; }
.card .label > .tip { flex: 0 0 auto; }
.card .value { font-size: 27px; font-weight: 650; margin-top: 6px; letter-spacing: -0.02em; }
.card .note { font-size: 12.5px; color: var(--muted); margin-top: 4px; }
.value.good { color: var(--good); }
.value.warn { color: var(--warn); }
.value.bad { color: var(--bad); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); }
th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { background: #eaf5ef; font-weight: 600; }
.legend { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--muted); margin-top: 10px; }
.note-block { font-size: 13.5px; color: var(--muted); margin-top: 10px; }
.badge {
  display: inline-block; font-size: 12px; padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--line); background: #f0f3f6; color: var(--muted); margin-right: 6px;
}
.badge.ok { background: #eaf5ef; border-color: #bfe0cd; color: var(--good); }
.badge.warn { background: #fbf2e3; border-color: #ebd7b0; color: var(--warn); }
.badge.bad { background: #fdecec; border-color: #e8b4b4; color: var(--bad); }
figure { margin: 0; }
figcaption { font-size: 13px; color: var(--muted); margin-top: 8px; }
.mrk-strip, .hw-strip {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 16px 0 14px;
}
.mrk-stat, .hw-stat, .bill-stat {
  background: #f7f9fb; border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px;
}
.mrk-stat .k, .hw-stat .k, .bill-stat .k {
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
}
.mrk-stat .v, .hw-stat .v { font-size: 24px; font-weight: 650; margin-top: 4px; letter-spacing: -0.02em; }
.bill-stat .v { font-size: 28px; font-weight: 650; margin-top: 6px; letter-spacing: -0.02em; }
.mrk-stat .v.good, .hw-stat .v.good, .bill-stat .v.good { color: var(--good); }
.hw-stat .v { font-size: 18px; line-height: 1.25; }
.mrk-stat .note, .hw-stat .note, .bill-stat .note { font-size: 12.5px; color: var(--muted); margin-top: 4px; }
.bill-strip {
  display: grid; grid-template-columns: 1fr auto 1fr auto 1fr minmax(150px, 0.95fr);
  gap: 10px; align-items: stretch; margin: 0;
}
.bill-arrow {
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 22px; font-weight: 600; padding: 0 2px;
}
@media (max-width: 980px) {
  .bill-strip { grid-template-columns: 1fr 1fr; }
  .bill-arrow { display: none; }
}
@media (max-width: 820px) {
  .bill-strip { grid-template-columns: 1fr; }
  .hw-strip { grid-template-columns: 1fr; }
}
.lang-switch { display: inline-flex; gap: 4px; margin-left: auto; }
.lang-switch button {
  border: 1px solid var(--line); background: #fff; color: var(--muted);
  border-radius: 6px; padding: 4px 10px; font: inherit; font-size: 12px;
  letter-spacing: 0.06em; cursor: pointer;
}
.lang-switch button.on { background: var(--ink); color: #fff; border-color: var(--ink); }
header .head-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.day-pick { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 8px 0 12px; font-size: 14px; }
.day-pick input { font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; }
.field { margin: 0 0 14px; }
.field label { display: flex; align-items: center; font-weight: 600; margin-bottom: 4px; }
.field input, .field select, .field textarea {
  width: 100%; max-width: 520px; font: inherit; padding: 8px 10px;
  border: 1px solid var(--line); border-radius: 6px; background: #fff;
}
.field input[type="file"] { max-width: 100%; }
.field .hint { font-size: 12.5px; color: var(--muted); margin-top: 4px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 24px; }
@media (max-width: 820px) { .grid-2 { grid-template-columns: 1fr; } }
.check { display: flex; align-items: flex-start; gap: 8px; margin: 10px 0; font-weight: 600; }
.check input { width: auto; margin-top: 4px; }
.btn {
  display: inline-block; background: var(--ink); color: #fff; border: 0;
  border-radius: 8px; padding: 10px 22px; font: inherit; font-weight: 650;
  letter-spacing: 0.02em; cursor: pointer;
}
.btn:hover { opacity: .92; }
.btn[disabled] { opacity: .55; cursor: wait; }
.nav-back { margin: 0 0 18px; }
.nav-back a { color: var(--accent); text-decoration: none; font-size: 14px; }
.nav-back a:hover { text-decoration: underline; }
progress { width: 100%; height: 10px; }
@media print {
  body { background: #fff; }
  .panel, .card { break-inside: avoid; }
  .lang-switch, .btn, .nav-back { display: none; }
}
"""

# Same behaviour as UC3.2 SEED: data-i18n, data-copy-sk/en, h1 titles, localStorage.
LANG_JS = r"""
(function () {
  function applyLang(lang) {
    window.UC32_LANG = lang;
    window.SOMES_LANG = lang;
    document.documentElement.lang = lang;
    var pack = {};
    var node = document.getElementById("i18n-pack");
    if (node) {
      try { pack = JSON.parse(node.textContent) || {}; } catch (err) { pack = {}; }
    }
    var t = pack[lang] || pack.en || {};
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      if (!el.getAttribute("data-sk")) el.setAttribute("data-sk", el.textContent);
      var key = el.getAttribute("data-i18n");
      if (lang === "en" && t[key]) el.textContent = t[key];
      if (lang === "sk") el.textContent = el.getAttribute("data-sk");
    });
    document.querySelectorAll("[data-copy-en]").forEach(function (el) {
      var next = lang === "en"
        ? (el.getAttribute("data-copy-en") || "")
        : (el.getAttribute("data-copy-sk") || "");
      if (!next) return;
      if (el.getAttribute("data-copy-html") === "1") el.innerHTML = next;
      else el.textContent = next;
    });
    var h1 = document.querySelector("h1[data-title-sk]");
    if (h1) {
      var next = lang === "en"
        ? (h1.getAttribute("data-title-en") || h1.textContent)
        : (h1.getAttribute("data-title-sk") || h1.textContent);
      h1.textContent = next;
      document.title = next;
    }
    document.querySelectorAll(".lang-switch button").forEach(function (btn) {
      btn.classList.toggle("on", btn.getAttribute("data-lang") === lang);
    });
    try {
      localStorage.setItem("uc32-lang", lang);
      localStorage.setItem("uc33-lang", lang);
    } catch (err) {}
    var day = document.getElementById("day-input");
    if (day && day.value && typeof window.SOMES_drawDay === "function") {
      window.SOMES_drawDay(day.value);
    }
  }
  window.applyLang = applyLang;
  window.applySomesLang = applyLang;
  document.querySelectorAll(".lang-switch button").forEach(function (btn) {
    btn.addEventListener("click", function () { applyLang(btn.getAttribute("data-lang")); });
  });
  var saved = "sk";
  try { saved = localStorage.getItem("uc32-lang") || localStorage.getItem("uc33-lang") || "sk"; } catch (err) {}
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { if (saved === "en") applyLang("en"); });
  } else if (saved === "en") {
    applyLang("en");
  }
})();
"""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def tip(text: str, i18n: str = "") -> str:
    if not text:
        return ""
    attr = f' data-i18n="{esc(i18n)}"' if i18n else ""
    return (
        f'<span class="tip" tabindex="0">?'
        f'<span class="tip-text"{attr}>{esc(text)}</span></span>'
    )


def labeled(label: str, hint: str = "", i18n: str = "", i18n_tip: str = "") -> str:
    text = f'<span data-i18n="{esc(i18n)}">{esc(label)}</span>' if i18n else esc(label)
    return f"{text}{tip(hint, i18n=i18n_tip)}"


def copy_el(sk: str, en: str, *, html_inner: bool = False, tag: str = "span") -> str:
    extra = ' data-copy-html="1"' if html_inner else ""
    inner = sk if html_inner else esc(sk)
    return (
        f"<{tag}{extra} data-copy-sk=\"{esc(sk)}\" data-copy-en=\"{esc(en)}\">{inner}</{tag}>"
    )


def lang_switch() -> str:
    return (
        '<div class="lang-switch" role="group" aria-label="Language">'
        '<button type="button" data-lang="sk" class="on">SK</button>'
        '<button type="button" data-lang="en">EN</button>'
        "</div>"
    )


def pack_script(en: dict[str, str]) -> str:
    payload = json.dumps({"en": en}, ensure_ascii=False)
    return f'<script type="application/json" id="i18n-pack">{payload}</script>'


def page_shell(*, title: str, title_en: str, kicker: str, sub: str, body: str, en: dict[str, str], extra_js: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{esc(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="head-row">
        <div>
          <p class="kicker" data-i18n="kicker">{esc(kicker)}</p>
          <h1 data-title-sk="{esc(title)}" data-title-en="{esc(title_en)}">{esc(title)}</h1>
          <p class="sub" data-i18n="sub">{esc(sub)}</p>
        </div>
        {lang_switch()}
      </div>
    </header>
    {body}
  </div>
  {pack_script(en)}
  <script>{LANG_JS}</script>
  {extra_js}
</body>
</html>
"""
