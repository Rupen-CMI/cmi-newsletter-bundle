"""Convert a rendered `templates/cmi_brief.html` brief into email-safe HTML.

The browser template leans on a <style> block, CSS grid/flexbox, and CSS custom
properties (`var(--navy)`, etc.) — Outlook and many other clients discard all of
that, so a sent brief can arrive unstyled or broken. This module re-emits the same
design as nested tables with literal colors inlined on every element.

    from emailify import emailify
    email_html = emailify(open(path, encoding="utf-8").read())

Parsing is by CSS class, so this file must be kept in sync with
templates/cmi_brief.html's class names — anything it doesn't recognize is
silently skipped rather than half-rendered.
"""
from __future__ import annotations

from bs4 import BeautifulSoup

SERIF = "'Roboto Slab',Georgia,'Times New Roman',serif"
SANS = "Roboto,'Segoe UI',Helvetica,Arial,sans-serif"

WIDTH = 660
PAD = "0 40px"

NAVY = "#002060"
GREEN = "#70AD47"
INK = "#212529"
PAPER = "#FFFFFF"
BG = "#EEF1F6"
MUTED = "#5B6470"
FAINT = "#8A93A3"
HAIR = "#E1E6EF"
HAIR_DK = "#20365F"

CMI_LOGO_URL = "https://www.coherentmarketinsights.com/images/indexv2/CMI-logo.svg"


# ------------------------------------------------------------- utilities --
def _inner(el) -> str:
    """Inner HTML of a tag, with <a> restyled for mail clients."""
    if el is None:
        return ""
    for a in el.find_all("a"):
        a["style"] = f"color:{GREEN};text-decoration:underline;"
    for b in el.find_all("b"):
        b["style"] = f"color:{NAVY};font-weight:600;"
    return el.decode_contents().strip()


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el is not None else ""


def _row(content: str, pad: str = PAD, bg: str = PAPER) -> str:
    return (
        f'<tr><td bgcolor="{bg}" style="background-color:{bg};padding:{pad};">'
        f"{content}</td></tr>"
    )


def _p(html: str, size="14px", color="#33383F", margin="0 0 16px") -> str:
    return (
        f'<p style="margin:{margin};font-family:{SANS};font-size:{size};'
        f'line-height:1.6;color:{color};mso-line-height-rule:exactly;">{html}</p>'
    )


def _spacer(h: int) -> str:
    return f'<div style="line-height:{h}px;font-size:{h}px;height:{h}px;">&nbsp;</div>'


def _hr(color: str = HAIR) -> str:
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td height="1" bgcolor="{color}" style="background-color:{color};'
        f'line-height:1px;font-size:1px;">&nbsp;</td></tr></table>'
    )


# --------------------------------------------------------------- sections --
def _masthead(soup) -> str:
    head = soup.select_one("header.masthead")
    if head is None:
        return ""
    k1 = _text(head.select_one(".k1"))
    k2 = _text(head.select_one(".k2"))
    tag = _inner(head.select_one(".tagstrip"))

    return f"""
<tr><td bgcolor="{PAPER}" style="background-color:{PAPER};padding:26px 40px 20px;border-bottom:1px solid {HAIR};">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
    <tr>
      <td align="left" valign="top">
        <img src="{CMI_LOGO_URL}" width="200" alt="Coherent Market Insights" style="display:block;max-width:200px;height:auto;">
      </td>
      <td align="right" valign="top" style="font-family:{SANS};line-height:1.6;">
        <span style="font-size:10.5px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:{GREEN};">{k1}</span><br>
        <span style="font-size:10px;color:{FAINT};font-weight:500;letter-spacing:.14em;">{k2}</span>
      </td>
    </tr>
  </table>
  {_spacer(14)}
  {_hr()}
  <div style="padding-top:14px;font-family:{SANS};font-size:12.5px;color:{MUTED};letter-spacing:.02em;">{tag}</div>
</td></tr>"""


def _intro(soup) -> str:
    out = []
    date = _text(soup.select_one(".datepill"))
    if date:
        out.append(
            f'<p style="margin:0 0 16px;font-family:{SANS};font-size:11px;font-weight:600;'
            f'letter-spacing:.14em;text-transform:uppercase;color:{GREEN};">{date}</p>'
        )
    hello = soup.select_one("h1.hello")
    if hello:
        out.append(
            f'<h1 style="margin:0 0 12px;font-family:{SERIF};font-weight:700;font-size:26px;'
            f'line-height:1.15;color:{NAVY};">{_text(hello)}</h1>'
        )
    lede = soup.select_one("p.lede")
    if lede:
        out.append(_p(_inner(lede), size="15px", color="#3A4048", margin="0 0 12px"))
    agent = soup.select_one("p.agentline")
    if agent:
        out.append(f"""
<table role="presentation" cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td width="3" bgcolor="{GREEN}" style="background-color:{GREEN};width:3px;line-height:1px;font-size:1px;">&nbsp;</td>
    <td style="padding-left:11px;font-family:{SANS};font-size:13px;line-height:1.6;color:{MUTED};">{_inner(agent)}</td>
  </tr>
</table>{_spacer(26)}""")
    return _row("".join(out), pad="32px 40px 0") if out else ""


def _top3(soup) -> str:
    box = soup.select_one(".top3")
    if box is None:
        return ""
    rows = [f'<tr><td colspan="2" height="2" bgcolor="{NAVY}" '
            f'style="background-color:{NAVY};line-height:2px;font-size:2px;">&nbsp;</td></tr>']
    for r in box.select(".row"):
        idx = _text(r.select_one(".idx"))
        txt = r.select_one(".txt")
        cat_el = txt.select_one(".cat") if txt else None
        cat = ""
        if cat_el:
            cat = (
                f'<span style="display:inline-block;font-family:{SANS};font-size:9.5px;font-weight:700;'
                f'letter-spacing:.1em;text-transform:uppercase;color:#fff;background-color:{NAVY};'
                f'border-radius:3px;padding:2px 7px;margin-right:9px;white-space:nowrap;">{_text(cat_el)}</span> '
            )
            cat_el.extract()
        rows.append(f"""
<tr>
  <td width="34" valign="top" style="padding:13px 12px 13px 0;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0">
      <tr><td width="22" height="22" align="center" valign="middle" bgcolor="{NAVY}"
              style="background-color:{NAVY};width:22px;height:22px;border-radius:3px;
                     font-family:{SANS};font-size:11px;font-weight:700;color:#fff;">{idx}</td></tr>
    </table>
  </td>
  <td valign="top" style="padding:13px 0;font-family:{SANS};font-size:14px;line-height:1.6;color:#33383F;">
    {cat}{_inner(txt)}
  </td>
</tr>
<tr><td colspan="2" height="1" bgcolor="{HAIR}" style="background-color:{HAIR};line-height:1px;font-size:1px;">&nbsp;</td></tr>""")
    table = (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
             f'{"".join(rows)}</table>')
    return _row(table, pad="0 40px 6px")


def _eyebrow(label: str) -> str:
    inner = f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td valign="middle" style="font-family:{SANS};font-size:11px;font-weight:700;letter-spacing:.16em;
        line-height:1.2;text-transform:uppercase;color:{NAVY};white-space:nowrap;padding:0 12px 0 0;">{label}</td>
    <td valign="middle" width="99%" style="width:99%;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr><td height="2" bgcolor="{GREEN}" style="height:2px;background-color:{GREEN};line-height:2px;font-size:2px;">&nbsp;</td></tr>
      </table>
    </td>
  </tr>
</table>"""
    return _row(inner, pad="42px 40px 18px")


def _kv_cell(h: str, v: str) -> str:
    return (f'<td width="50%" valign="top" bgcolor="{PAPER}" '
            f'style="background-color:{PAPER};padding:13px 15px;border:1px solid {HAIR};">'
            f'<div style="font-family:{SANS};font-size:10px;font-weight:900;letter-spacing:.14em;'
            f'text-transform:uppercase;color:{NAVY};margin-bottom:6px;">{h}</div>'
            f'<div style="font-family:{SANS};font-size:13px;line-height:1.45;color:{INK};">{v}</div></td>')


def _prospect(card) -> str:
    flag = _text(card.select_one(".flag"))
    name = _text(card.select_one("h2"))
    para = card.find("p", recursive=False) or card.find("p")
    cells = card.select(".grid2 .cell")

    body = [
        f'<span style="display:inline-block;font-family:{SANS};font-size:10px;font-weight:700;'
        f'letter-spacing:.12em;text-transform:uppercase;color:#fff;background-color:{NAVY};'
        f'padding:4px 10px;border-radius:3px;margin-bottom:13px;">{flag}</span>',
        f'<h2 style="margin:14px 0 10px;font-family:{SERIF};font-weight:700;font-size:23px;'
        f'line-height:1.2;color:{NAVY};">{name}</h2>',
        _p(_inner(para)),
    ]
    if cells:
        tds = "".join(_kv_cell(_text(c.select_one(".h")), _text(c.select_one(".v"))) for c in cells)
        body.append(
            f'{_spacer(4)}<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            f'border="0" style="border-collapse:collapse;"><tr>{tds}</tr></table>{_spacer(6)}'
        )

    return f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td width="4" bgcolor="{GREEN}" style="background-color:{GREEN};width:4px;line-height:1px;font-size:1px;">&nbsp;</td>
    <td bgcolor="{PAPER}" style="background-color:{PAPER};border:1px solid {HAIR};border-left:0;padding:22px 24px;">
      {"".join(body)}
    </td>
  </tr>
</table>{_spacer(20)}"""


def _story(el) -> str:
    out = [
        f'<h3 style="margin:0 0 9px;font-family:{SERIF};font-weight:500;font-size:19px;'
        f'line-height:1.25;color:{NAVY};">{_text(el.select_one("h3"))}</h3>'
    ]
    for p in el.find_all("p"):
        out.append(_p(_inner(p), margin="0"))
    return (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<tr><td style="padding:6px 0 18px;">{"".join(out)}</td></tr></table>{_spacer(18)}')


def _analyst(el) -> str:
    tag = _text(el.select_one(".tag"))
    items = "".join(
        f'<tr><td width="14" valign="top" style="padding:0 8px 10px 0;">'
        f'<div style="width:5px;height:5px;margin-top:9px;background-color:{GREEN};border-radius:1px;"></div></td>'
        f'<td valign="top" style="padding:0 0 10px;font-family:{SANS};font-size:14px;line-height:1.6;color:#33383F;">'
        f'{_inner(li)}</td></tr>'
        for li in el.select("ul li")
    )
    prov = el.select_one(".prov")
    prov_html = ""
    if prov:
        src = _text(prov.select_one(".src"))
        spans = prov.find_all("span", recursive=False)
        date = ""
        for s in spans:
            if not (s.get("class") or []):
                date = s.get_text(strip=True)
        verified = prov.select_one(".verified")
        bits = f'<span style="color:{NAVY};font-weight:600;">{src}</span>&nbsp;&middot;&nbsp;<span style="color:{FAINT};">{date}</span>'
        # This row is rebuilt from scratch rather than copied, so any anchor the
        # template added has to be re-emitted explicitly -- otherwise the citation
        # link is clickable in the browser file and silently missing in the email.
        link = prov.select_one("a.srclink")
        if link and link.get("href"):
            bits += (f'&nbsp;&middot;&nbsp;<a href="{link["href"]}" target="_blank" '
                     f'style="color:{NAVY};text-decoration:underline;">{_text(link)}</a>')
        if verified:
            bits += (f'&nbsp;&nbsp;<span style="color:{GREEN};border:1px solid rgba(112,173,71,.45);'
                      f'border-radius:3px;padding:2px 7px;background-color:rgba(112,173,71,.09);">{_text(verified)}</span>')
        prov_html = (f'<div style="margin-top:12px;font-family:{SANS};font-size:10.5px;letter-spacing:.06em;'
                      f'text-transform:uppercase;color:{FAINT};">{bits}</div>')
    return f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr><td style="padding-top:16px;border-top:1px solid {HAIR};">
    <div style="font-family:{SANS};font-size:10.5px;font-weight:700;letter-spacing:.16em;
        text-transform:uppercase;color:{NAVY};margin:0 0 12px;">{tag}</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">{items}</table>
    {prov_html}
  </td></tr>
</table>{_spacer(6)}"""


def _seccta(el) -> str:
    q = _text(el.select_one(".q"))
    sub = _text(el.select_one(".sub"))
    btn = el.select_one("a.btn")
    href = btn.get("href", "#") if btn else "#"
    label = _text(btn) if btn else "Learn more"
    return f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr><td align="center" bgcolor="#F3F8F0" style="background-color:#F3F8F0;border:1px solid rgba(112,173,71,.30);
      border-top:3px solid {GREEN};padding:24px 26px;">
    <p style="margin:0 0 6px;font-family:{SERIF};font-size:16px;line-height:1.4;font-weight:500;color:{NAVY};">{q}</p>
    <p style="margin:0 0 18px;font-family:{SANS};font-size:13px;line-height:1.55;color:{MUTED};">{sub}</p>
    <a href="{href}" target="_blank" style="display:inline-block;background-color:{GREEN};color:#fff;
       text-decoration:none;font-family:{SANS};font-weight:700;font-size:14px;letter-spacing:.02em;
       padding:13px 28px;border-radius:4px;">{label} &rarr;</a>
  </td></tr>
</table>{_spacer(8)}"""


def _caps(el) -> str:
    h3 = _inner(el.select_one("h3"))
    sub = _inner(el.select_one(".capsub"))

    stats = el.select(".capgrid .cap")
    stat_rows = ""
    for i in range(0, len(stats), 2):
        tds = ""
        for c in stats[i:i + 2]:
            n = c.select_one(".n")
            sfx = n.select_one(".sfx") if n else None
            sfx_txt = _text(sfx)
            n_txt = n.get_text(strip=True).replace(sfx_txt, "") if n else ""
            tds += (
                f'<td width="50%" valign="top" style="padding:0 12px 26px 0;">'
                f'<div style="font-family:{SERIF};font-size:32px;line-height:1;color:#fff;">'
                f'{n_txt}<span style="font-size:17px;color:{GREEN};">{sfx_txt}</span></div>'
                f'<div style="margin-top:9px;font-family:{SANS};font-size:12px;line-height:1.4;'
                f'color:#8FA0C0;">{_text(c.select_one(".l"))}</div></td>'
            )
        if len(stats[i:i + 2]) == 1:
            tds += '<td width="50%">&nbsp;</td>'
        stat_rows += f"<tr>{tds}</tr>"

    cards = el.select(".capcard")
    card_rows = ""
    for i in range(0, len(cards), 2):
        pair = cards[i:i + 2]
        tds = ""
        for c in pair:
            ps = "".join(
                f'<p style="margin:0 0 8px;font-family:{SANS};font-size:12px;line-height:1.6;'
                f'color:#8FA0C0;">{_inner(p)}</p>' for p in c.find_all("p")
            )
            tds += (
                f'<td width="50%" valign="top" style="padding:0 13px 22px 0;">'
                f'<div style="font-family:{SANS};font-size:9.5px;font-weight:700;letter-spacing:.16em;'
                f'text-transform:uppercase;color:{GREEN};margin-bottom:8px;">{_text(c.select_one(".cl"))}</div>'
                f'<h4 style="margin:0 0 7px;font-family:{SANS};font-weight:700;font-size:13.5px;'
                f'line-height:1.35;color:#fff;">{_text(c.select_one("h4"))}</h4>{ps}</td>'
            )
        if len(pair) == 1:
            tds += '<td width="50%">&nbsp;</td>'
        card_rows += f"<tr>{tds}</tr>"

    divider = (
        f"{_spacer(24)}"
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td height="1" bgcolor="{HAIR_DK}" style="height:1px;background-color:{HAIR_DK};'
        f'line-height:1px;font-size:1px;">&nbsp;</td></tr></table>{_spacer(24)}'
    )

    logos = el.select(".lw-grid .lw-tile img")
    logo_block = ""
    if logos:
        tds = ""
        for i in range(0, len(logos), 5):
            row = logos[i:i + 5]
            tds += "<tr>" + "".join(
                f'<td width="20%" align="center" valign="middle" bgcolor="#fff" '
                f'style="background-color:#fff;padding:8px;height:64px;">'
                f'<img src="{img.get("src")}" alt="{img.get("alt", "")}" style="max-height:48px;max-width:100%;"></td>'
                for img in row
            ) + "</tr>"
        logo_block = (
            f"{divider}"
            f'<div style="font-family:{SANS};font-size:9.5px;font-weight:700;letter-spacing:.16em;'
            f'text-transform:uppercase;color:{GREEN};margin-bottom:14px;">Our Existing Clientele</div>'
            f'<table role="presentation" width="100%" cellpadding="4" cellspacing="0" border="0">{tds}</table>'
        )

    return f"""{_spacer(6)}
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr><td bgcolor="{NAVY}" style="background-color:{NAVY};padding:32px 32px 26px;border-radius:4px;">
    <h3 style="margin:0 0 10px;font-family:{SERIF};font-weight:700;font-size:22px;line-height:1.25;
        color:#fff;">{h3}</h3>
    <p style="margin:0;font-family:{SANS};font-size:13px;line-height:1.6;color:#9DACC8;">{sub}</p>
    {divider}
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">{stat_rows}</table>
    {divider}
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">{card_rows}</table>
    {logo_block}
  </td></tr>
</table>{_spacer(6)}"""


def _testimonials(el) -> str:
    cards = el.select(".tcard")
    rows = ""
    for c in cards:
        quote = _text(c.select_one(".tq"))
        role = _text(c.select_one(".trole"))
        co = _text(c.select_one(".tco"))
        rows += f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td width="3" bgcolor="{GREEN}" style="background-color:{GREEN};width:3px;line-height:1px;font-size:1px;">&nbsp;</td>
    <td bgcolor="{PAPER}" style="background-color:{PAPER};border:1px solid {HAIR};border-left:0;padding:18px 20px;">
      <p style="margin:0 0 14px;font-family:{SERIF};font-size:14px;line-height:1.55;color:#33383F;font-style:italic;">{quote}</p>
      <div style="font-family:{SANS};font-size:12px;color:{MUTED};">{role}</div>
      <div style="font-family:{SANS};font-size:12.5px;font-weight:600;color:{NAVY};">{co}</div>
    </td>
  </tr>
</table>{_spacer(12)}"""
    return rows


def _certs(el) -> str:
    imgs = el.select(".c-grid .c-tile img")
    if not imgs:
        return ""
    tds = ""
    for i in range(0, len(imgs), 3):
        row = imgs[i:i + 3]
        tds += "<tr>" + "".join(
            f'<td width="33%" align="center" valign="middle" bgcolor="#fff" '
            f'style="background-color:#fff;border:1px solid {HAIR};padding:8px;height:64px;">'
            f'<img src="{img.get("src")}" alt="{img.get("alt", "")}" style="max-height:48px;max-width:100%;"></td>'
            for img in row
        ) + "</tr>"
    return f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr><td style="padding-top:22px;border-top:2px solid {NAVY};">
    <div style="font-family:{SANS};font-size:9.5px;font-weight:700;letter-spacing:.16em;
        text-transform:uppercase;color:{NAVY};margin-bottom:14px;">Credibility &amp; certifications</div>
    <table role="presentation" width="100%" cellpadding="4" cellspacing="0" border="0">{tds}</table>
  </td></tr>
</table>"""


def _footer(soup) -> str:
    foot = soup.select_one("footer.foot")
    if foot is None:
        return ""
    desc = _text(foot.select_one(".cmi-desc"))
    assembled = _text(foot.select_one(".assembled"))
    sign = _text(foot.select_one(".sign"))
    meta = foot.select_one(".meta")
    for a in (meta.find_all("a") if meta else []):
        a["style"] = f"color:{FAINT};text-decoration:underline;"

    btns = ""
    primary = foot.select_one("a.cta")
    if primary:
        btns += (
            f'<td style="padding:0 6px;"><a href="{primary.get("href", "#")}" target="_blank" '
            f'style="display:inline-block;background-color:{GREEN};color:#fff;text-decoration:none;'
            f'font-family:{SANS};font-weight:700;font-size:13.5px;padding:12px 26px;border-radius:4px;">'
            f'{_text(primary)}</a></td>'
        )
    secondary = foot.select_one("a.cta2")
    if secondary:
        btns += (
            f'<td style="padding:0 6px;"><a href="{secondary.get("href", "#")}" target="_blank" '
            f'style="display:inline-block;color:#DDE4F0;text-decoration:none;font-family:{SANS};'
            f'font-weight:600;font-size:13.5px;padding:11px 24px;border-radius:4px;'
            f'border:1px solid rgba(221,228,240,.30);">{_text(secondary)}</a></td>'
        )

    return f"""
<tr><td bgcolor="{NAVY}" align="center" style="background-color:{NAVY};padding:30px 40px 34px;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center"><tr><td
     bgcolor="#fff" style="background-color:#fff;padding:12px 18px;border-radius:4px;">
     <img src="{CMI_LOGO_URL}" width="160" alt="Coherent Market Insights" style="display:block;max-width:160px;height:auto;">
  </td></tr></table>
  <p style="margin:14px auto 18px;font-family:{SANS};font-size:12px;line-height:1.6;color:#B9C4DA;max-width:340px;">{desc}</p>
  <p style="margin:0 0 16px;font-family:{SANS};font-size:10.5px;letter-spacing:.14em;
     text-transform:uppercase;color:#8FA0C0;">{assembled}</p>
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center"><tr>{btns}</tr></table>
  <p style="margin:20px 0 0;font-family:{SERIF};font-style:italic;font-size:13.5px;color:#DDE4F0;">{sign}</p>
  <p style="margin:6px 0 0;font-family:{SANS};font-size:10.5px;letter-spacing:.1em;
     text-transform:uppercase;color:#7C8CAC;">{_inner(meta)}</p>
</td></tr>"""


# ------------------------------------------------------------------ main --
def emailify(html: str) -> str:
    """Convert a rendered `cmi_brief.html` output to email-safe table-based HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = _text(soup.select_one("title"))

    rows = [_masthead(soup), _intro(soup), _top3(soup)]

    main = soup.select_one("main.body")
    body_parts: list[str] = []
    for el in (main.find_all(recursive=False) if main else []):
        cls = el.get("class") or []
        if el.name in ("p", "h1") and any(c in cls for c in ("datepill", "lede", "agentline", "hello")):
            continue
        if "top3" in cls:
            continue
        if "eyebrow" in cls:
            if body_parts:
                rows.append(_row("".join(body_parts)))
                body_parts = []
            rows.append(_eyebrow(_text(el.select_one(".lbl"))))
        elif "prospect" in cls:
            body_parts.append(_prospect(el))
        elif "story" in cls:
            body_parts.append(_story(el))
        elif "analyst" in cls:
            body_parts.append(_analyst(el))
        elif "seccta" in cls:
            body_parts.append(_seccta(el))
        elif "caps" in cls:
            body_parts.append(_caps(el))
        elif "tgrid" in cls:
            body_parts.append(_testimonials(el))
        elif "certs" in cls:
            body_parts.append(_certs(el))
    if body_parts:
        rows.append(_row("".join(body_parts)))
    rows.append(_row(_spacer(30)))
    rows.append(_footer(soup))

    sheet = "".join(r for r in rows if r)

    return f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="x-apple-disable-message-reformatting" />
<meta name="color-scheme" content="light" />
<meta name="supported-color-schemes" content="light" />
<title>{title}</title>
<!--[if mso]><xml><o:OfficeDocumentSettings>
<o:AllowPNG/><o:PixelsPerInch>96</o:PixelsPerInch>
</o:OfficeDocumentSettings></xml><![endif]-->
<style type="text/css">
  :root {{ color-scheme: light; supported-color-schemes: light; }}
  body,table,td,p,h1,h2,h3,h4,a {{ -webkit-text-size-adjust:100%; -ms-text-size-adjust:100%; }}
  table {{ border-collapse:collapse; mso-table-lspace:0pt; mso-table-rspace:0pt; }}
  img {{ border:0; outline:none; text-decoration:none; -ms-interpolation-mode:bicubic; }}
  [data-ogsc] .sheet, [data-ogsb] .sheet {{ background-color:{BG} !important; }}
  @media only screen and (max-width:600px) {{
    .sheet {{ width:100% !important; }}
    .sheet td {{ padding-left:20px !important; padding-right:20px !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background-color:{BG};">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
       bgcolor="{BG}" style="background-color:{BG};">
  <tr><td align="center" style="padding:32px 12px 64px;">
    <table role="presentation" class="sheet" width="{WIDTH}" cellpadding="0" cellspacing="0" border="0"
           bgcolor="{PAPER}" style="width:{WIDTH}px;max-width:{WIDTH}px;background-color:{PAPER};">
      {sheet}
    </table>
  </td></tr>
</table>
</body>
</html>"""


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        out = emailify(f.read())
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{src} -> {dst} ({len(out):,} bytes)")
