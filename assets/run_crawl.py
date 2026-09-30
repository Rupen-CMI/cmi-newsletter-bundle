"""Crawl a list of company domains with smartcrawl and dump one JSON per company.

    python run_crawl.py --domains vvfltd.com paytm.com entrust.co.in --out /home/intel

Runs on the GPU box because smartcrawl wants Linux, a browser tier, and curl_cffi --
none of which are set up on the Windows laptop. The output is plain JSON, so it comes
back over `jl download` and feeds company_intel.py locally.

Crawls are sequential by default: they are LLM-extraction heavy, and running four at
once against one DeepSeek key mostly buys rate-limit errors.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backend_shim  # noqa: E402

backend_shim.install()


async def crawl_one(domain: str, out_dir: str, mode: str, max_pages: int) -> dict:
    from smartcrawl import smart_crawl

    t0 = time.perf_counter()
    slug = "".join(c for c in domain if c.isalnum() or c in ".-").replace(".", "_")
    path = os.path.join(out_dir, f"{slug}.json")
    try:
        res = await smart_crawl(domain, mode=mode, max_pages=max_pages)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        data = res.get("data") or {}
        comp = ((data.get("relationships") or {}).get("competitors") or [])
        desc = ((data.get("company") or {}).get("description") or "")[:70]
        return {"domain": domain, "ok": True, "secs": round(time.perf_counter() - t0),
                "path": path, "competitors": len(comp), "desc": desc,
                "error": res.get("error", "")}
    except Exception as e:  # noqa: BLE001 -- one bad domain must not lose the batch
        return {"domain": domain, "ok": False, "secs": round(time.perf_counter() - t0),
                "error": f"{type(e).__name__}: {e}"}


async def main_async(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for d in args.domains:
        print(f"--- crawling {d} ...", flush=True)
        r = await crawl_one(d, args.out, args.mode, args.max_pages)
        rows.append(r)
        status = "ok" if r["ok"] else "FAIL"
        extra = r.get("error") or f"{r.get('competitors', 0)} competitors | {r.get('desc', '')}"
        print(f"    {status}  {r['secs']}s  {extra}", flush=True)

    print("\n" + "=" * 62)
    for r in rows:
        print(f"{r['domain']:<22}{'ok' if r['ok'] else 'FAIL':<6}{r['secs']:>5}s  "
              f"{r.get('error') or str(r.get('competitors', 0)) + ' competitors'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", default="/home/intel")
    ap.add_argument("--mode", default="company",
                    help="smartcrawl mode; 'company' targets About/Products/Solutions pages "
                         "instead of crawling the whole news archive")
    ap.add_argument("--max-pages", type=int, default=25)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
