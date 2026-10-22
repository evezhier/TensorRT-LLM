# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a chart of measurements from the same decoder GPU pair."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TEMPLATE = r"""<!doctype html>
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;padding:20px;color:#152737;background:#fff;font:15px/1.5 system-ui,sans-serif}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:18px}select{font:inherit;padding:7px 10px;border:1px solid #8c9caa;border-radius:6px;background:#fff;color:inherit}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}.key:before{content:"";display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--color);margin-right:6px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.panel{padding:18px;background:#f5f7f9;border-radius:10px}.panel h2{font-size:16px;margin:0 0 4px}.unit{font-size:12px;color:#536574;margin-bottom:14px}.row{margin:12px 0 17px}.labels{display:flex;justify-content:space-between;gap:10px;font-size:13px;margin-bottom:5px}.labels strong{white-space:nowrap;font-variant-numeric:tabular-nums}.barback{height:18px;background:#e3e9ee;border-radius:3px}.bar,.hitbar{height:100%;border-radius:3px;background:var(--color)}.note{font-size:12px;color:#536574;margin-top:16px}.hitvalue{font-size:34px;font-weight:650;letter-spacing:-1px;margin:16px 0 8px}.small{font-size:12px;color:#536574}select:focus-visible{outline:3px solid #6847c7;outline-offset:3px}
@media(max-width:580px){body{padding:12px}.grid{grid-template-columns:1fr;gap:12px}.panel{padding:14px}.controls{gap:10px}}
</style>
<div class="controls"><label for="batch">Active decoder requests</label><select id="batch"><option value="1">1</option><option value="96">96</option><option value="144">144</option><option value="1204">1,204</option></select><div class="legend"><span class="key" style="--color:#176b98">Offload disabled</span><span class="key" style="--color:#b74b20">Offload enabled</span></div></div>
<div id="charts" class="grid" aria-live="polite"></div>
<div class="note">32,768 input / 160 output tokens · 1 prefill worker / 1 decoder · Same two decoder B200 GPUs · No speculative decoding. All KV transfers complete before each fixed batch starts. Decode throughput uses 158 complete intervals between the 159 decode iteration starts.</div>
<script>
const data=__DATA__;
const metrics=[['tps','Decoder throughput','decode tokens/s · iteration window',1],['gpu_ms','GPU forward','milliseconds · p50',2],['cadence_ms','Decode iteration cadence','milliseconds · p50',2]];
let selected='1204';
function accepted(s){return ['1','96','144','1204'].includes(s?.batch)?s.batch:null}
function render(){
 document.getElementById('batch').value=selected;
 const rows=data.rows.filter(r=>String(r.concurrency)===selected);
 const panels=metrics.map(([key,title,unit,digits])=>{
  const panel=document.createElement('section');panel.className='panel';
  const h=document.createElement('h2');h.textContent=title;panel.append(h);
  const u=document.createElement('div');u.className='unit';u.textContent=unit;panel.append(u);
  const scale=Math.max(...rows.map(r=>r[key]))*1.08;
  rows.forEach(r=>{
   const label=r.mode==='resident'?'Disabled':'Enabled';
   const value=r[key].toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits});
   const row=document.createElement('div');row.className='row';row.style.setProperty('--color',r.mode==='resident'?'#176b98':'#b74b20');
   const labels=document.createElement('div');labels.className='labels';
   const name=document.createElement('span');name.textContent=label;const strong=document.createElement('strong');strong.textContent=value;labels.append(name,strong);row.append(labels);
   const back=document.createElement('div');back.className='barback';back.setAttribute('role','img');back.setAttribute('aria-label',label+': '+value+' '+unit);
   const bar=document.createElement('div');bar.className='bar';bar.style.width=(100*r[key]/scale)+'%';back.append(bar);row.append(back);panel.append(row);
  });return panel;
 });
 const cache=document.createElement('section');cache.className='panel';
 const h=document.createElement('h2');h.textContent='Sparse GPU cache hit rate';cache.append(h);
 const unit=document.createElement('div');unit.className='unit';unit.textContent='All offload trials, including warmup · GPU cache factor 2.0';cache.append(unit);
 const value=document.createElement('div');value.className='hitvalue';value.textContent=data.hit_rate_percent.toFixed(2)+'%';cache.append(value);
 const back=document.createElement('div');back.className='barback';back.setAttribute('role','img');back.setAttribute('aria-label',value.textContent+' hits');
 const bar=document.createElement('div');bar.className='hitbar';bar.style.setProperty('--color','#22735b');bar.style.width=data.hit_rate_percent+'%';back.append(bar);cache.append(back);
 const counts=document.createElement('p');counts.className='small';counts.textContent=data.hits.toLocaleString('en-US')+' hits / '+(data.hits+data.misses).toLocaleString('en-US')+' selected units. Both decoder ranks combined.';cache.append(counts);
 panels.push(cache);document.getElementById('charts').replaceChildren(...panels);
}
const initial=accepted(window.openai?.widgetState);if(initial)selected=initial;
document.getElementById('batch').addEventListener('change',e=>{selected=e.target.value;render();window.openai?.setWidgetState?.({...window.openai?.widgetState,version:1,batch:selected});});
window.addEventListener('openai:set_globals',e=>{const value=accepted(e.detail?.globals?.widgetState);if(value&&value!==selected){selected=value;render();}});
render();
</script></html>"""


def main() -> None:
    report = json.loads((ROOT / "fix-analysis.json").read_text())
    rows = []
    offload = None
    for run in report["runs"]:
        if "fallback-p1" not in run["directory"]:
            continue
        mode = run["config"]["mode"]
        if mode == "offload":
            offload = run
        for trial in run["trials"]:
            if trial["trial"] not in ("100-c1", "101-c96", "102-c144", "103-c1204"):
                continue
            rows.append({
                "mode": mode, "concurrency": trial["command"]["concurrency"],
                "tps": trial["scheduled_tokens_per_iteration_window_second"],
                "gpu_ms": trial["generation_forward_event_ms"]["p50"],
                "cadence_ms": trial["decode_step_cadence_ms"]["p50"],
            })
    assert len(rows) == 8 and offload and offload["finished"]
    rows.sort(key=lambda r: (r["concurrency"], r["mode"] != "resident"))
    data = {
        "rows": rows, "hits": offload["total_sparse_hits"],
        "misses": offload["total_sparse_misses"],
        "hit_rate_percent": offload["sparse_hit_rate_percent"],
    }
    assert data["hit_rate_percent"] is not None
    (ROOT / "fix-visualization-data.json").write_text(json.dumps(data, indent=2) + "\n")
    html = TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (ROOT / "fix-comparison.html").write_text(html)
    print({"html_bytes": len(html.encode()), "rows": len(rows)})


if __name__ == "__main__":
    main()
