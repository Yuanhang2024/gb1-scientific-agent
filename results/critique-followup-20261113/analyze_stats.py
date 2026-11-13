# 批判性审视跟进统计：首答候选与高分变体重合率、假设锚点转移成功率。
# 只读取已保存的实验记录与官方S1，不发起任何模型请求。
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AMINO = set("ACDEFGHIKLMNPQRSTVWY")
AA4 = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY]{4})\b")

# 4字母、全由氨基酸字母组成的常见英文词，避免把正文当锚点。
STOP = set("""WHEN WHAT HIGH SAME READ GAIN MATE LINE WELL LESS MAIN MEAN NEAR WIDE WILL
THAT THIS THEN THEY WITH HAVE HERE EACH BOTH SAID MAKE TAKE GAVE GIVE TIDE SIDE
RISE RIDE WAGE WAVE PAGE PANE PINE DINE WINE TWIN TWICE WIRES WIRED OTHER THEIR
THEM THESE THOSE FROM FORM TEAM TEAR TERM TEST TEXT THAT THAN SOME SITE SINS
SING RANG RAIN HAIR HAND HANG HARD HATE HEAD HEAL HEAR HEAT HELD HELL HELP HENS
HERD HERO HIDE HIKE HILL HINT HIRE HOLD HOLE HOME HOPE HOST HOUR HUGE HUNT HURT
IDEA INTO IRON ITEM JOINS JUST KEEN KEEP KEPT KIND KNEW KNOW LACK LADY LAID LAIR
LAND LANE LAST LATE LEAD LEAN LEFT LEND LENT LESS LIES LIFT LIKE LIME LIND LINE
LINK LIST LIVE LOAD LOAN LOCK LOSE LOSS LOST LOUD LOVE LUCK MADE MAIL MAID MALE
MANY MAPS MARE MARK MASS MAST MATE MEAL MEAT MEET MELT MEND MENU MERE MESS MILD
MILE MILK MIND MINE MINT MISS MIST MODE MOOD MOON MORE MOST MOTH MOVE MUCH MULE
MUST NAIL NAME NAVY NEAR NEAT NECK NEED NEST NEWS NEXT NICE NINE NONE NOON NOSE
NOTE NOUN OATH OBEY ODDS ONCE ONLY ONTO OPEN ORAL OURS OVAL OVEN OVER PACE PACK
PAID PAIL PAIN PAIR PALE PALM PAPA PARK PART PASS PAST PATH PEAK PEAL PEAR PEAS
PEEK PEER PENS PENT PERK PEST PICK PIER PINE PINK PINT PIPE PITS PLAN PLAY PLEA
PLOT PLOW PLUG PLUS POEM POET POLL POND POOL POOR PORT POSE POST POUR PRAM PRAY
PREP PREY PROM PROP PROS PROVE PULL PUMP PUNK PURE PUSH QUIT QUITE RACE RACK
RAFT RAGE RAIL RAKE RAMP RANK RANT RARE RASH RATE RAVE READ REAL REAR REED REEF
REEK REIN RELY RENT REST RICE RICH RIDE RIFT RIME RING RINK RIOT RISE RISK RITE
ROAD ROAM ROAR ROCK RODE ROLE ROLL ROOF ROOM ROOT ROPE ROSE ROUT ROVE RUBY RUDE
RUIN RULE RUNG RUSH RUST SAFE SAGE SAID SAIL SAKE SALE SALT SAME SAND SANE SANG
SANK SAVE SCAR SEAL SEAM SEAT SEED SEEK SEEM SEEN SEES SELF SELL SEND SENT SETS
SEWN SHAG SHAM SHED SHIM SHIN SHIP SHOD SHOE SHOP SHOT SHOW SHUN SHUT SICK SIDE
SIFT SIGH SIGN SILK SILL SILO SINK SINS SIRE SITS SITE SIZE SKID SKIM SKIN SKIP
SKIS SLAB SLAM SLAP SLAT SLAV SLAW SLED SLEW SLID SLIM SLIP SLIT SLOB SLOE SLOG
SLOT SLOW SLUG SLUM SMUG SNAG SNAP SNIK SNOB SNOT SNOW SNUB SNAP SOAK SOAP SOAR
SOCK SODA SOFA SOFT SOIL SOLD SOLE SOLO SOME SONG SOON SOOT SORE SORT SOUL SOUP
SOUR SOWN SPAN SPAR SPAS SPAT SPAY SPED SPIN SPIT SPOT SPRY SPUD SPUN SPUR STAB
STAG STAR STAY STEM STEP STEW STIR STOP STOW STUB STUD STUN SUCH SUCK SUIT SULK
SURE SWAB SWAG SWAM SWAN SWAP SWAT SWAY SWIG SWIM TAIL TAKE TALC TALE TALK TALL
TAME TANK TAPE TAPS TARE TARP TART TASK TEAK TEAL TEAM TEAR TEEM TEEN TELL TEMP
TEND TENT TERM TEST TEXT THAN THAT THAW THEE THEM THEN THEY THIN THIS THUS TIDE
TIDY TIED TIER TIES TILE TILL TILT TIME TINE TINS TINT TINY TIPS TIRE TOAD TOGO
TOIL TOLD TOLL TOMB TONE TONS TOOK TOOL TORE TORN TOSS TOTE TOTS TOUR TOUT TOWN
TRAG TRAM TRAP TRAY TREE TRIM TRIO TRIP TROD TROT TRUE TSAR TUBA TUBE TUCK TUFT
TUNA TUNE TURF TURN TUSK TWIG TWIN TWIT TYPE UNIT UPON URGE USED USER VAIL VAIN
VARY VASE VAST VEAL VEER VEIL VEIN VEND VENT VERB VERY VEST VETO VIAL VIBE VICE
VIEW VILE VINE VISA VOID VOLT VOTE WACK WADE WAGE WAIF WAIL WAIT WAKE WALK WALL
WAND WANE WANT WARD WARE WARM WARN WART WARY WASH WASP WAVE WAVY WAYS WEAK WEAN
WEAR WEED WEEK WEEP WELD WELL WENT WEPT WERE WEST WHAT WHEN WHET WHEY WHIM WHIP
WHIR WHIT WHIZ WICK WIDE WIFE WIGS WILD WILE WILL WILT WIMP WIND WINE WING WINK
WINS WIPE WIRE WIRY WISE WISH WISS WITH WITS WOKE WOLF WOMB WONT WOOD WOOL WORD
WORE WORK WORM WORN WRAP WREN YAKS YARD YARN YAWN YEAR YELL YOGA YOUR ZEAL ZERO
ZEST ZINC ZONE""".split())


def load_global_fitness():
    """官方S1全表：149,361条唯一变体与实测fitness。"""
    source = ROOT / "data" / "raw" / "elife-16965-supp1-v4.xlsx"
    table = pd.read_excel(source, sheet_name="Supplementary Dataset 1")
    variants = table["Variants"].astype(str).str.upper().str.strip()
    fitness = pd.to_numeric(table["Fitness"], errors="coerce")
    keep = variants.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{4}") & fitness.notna() & np.isfinite(fitness) & (fitness >= 0)
    frame = pd.DataFrame({"variant": variants[keep], "fitness": fitness[keep].astype(float)})
    frame = frame.groupby("variant", as_index=False)["fitness"].first()
    assert len(frame) == 149361, f"unexpected S1 size {len(frame)}"
    return frame


def first_answer_variants(events_path):
    """每个(method, round)的首次回答候选（纠正前的原始响应）。"""
    if not events_path.exists():
        return []
    rows = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("kind") != "llm_response" or obj.get("attempt") != 1:
            continue
        try:
            body = json.loads(obj["response"]["text"])
            variants = [p["variant"] for p in body.get("proposals", [])]
        except Exception:
            continue
        rows.append({"method": obj["method"], "round": obj["round"], "variants": variants})
    return rows


def overlap_stats(global_fit, runs):
    """统计①：首答候选落入全局top-1%/top-5%的比率，并与均匀抽样期望比较。"""
    q99 = float(global_fit.fitness.quantile(0.99))
    q95 = float(global_fit.fitness.quantile(0.95))
    top10 = set(global_fit.nlargest(10, "fitness").variant)
    per_run, agg = [], {"llm": [0, 0, 0], "knowledge": [0, 0, 0]}
    for run_label, rows in runs:
        for row in rows:
            variants = [v for v in row["variants"] if v in global_fit.variant.values]
            n = len(variants)
            if n == 0:
                continue
            hit99 = sum(global_fit.set_index("variant").fitness[v] >= q99 for v in variants)
            hit95 = sum(global_fit.set_index("variant").fitness[v] >= q95 for v in variants)
            hit10 = sum(v in top10 for v in variants)
            per_run.append({"run": run_label, "method": row["method"], "round": row["round"],
                            "first_answer_n": n, "in_top1pct": int(hit99), "in_top5pct": int(hit95),
                            "in_global_top10": int(hit10)})
            bucket = agg.setdefault(row["method"], [0, 0, 0])
            bucket[0] += hit99; bucket[1] += hit95; bucket[2] += n
    summary = {}
    for method, (h99, h95, n) in agg.items():
        if n:
            summary[method] = {"first_answer_candidates": n,
                               "top1pct_rate": round(h99 / n, 4), "uniform_expectation_top1pct": 0.01,
                               "top5pct_rate": round(h95 / n, 4), "uniform_expectation_top5pct": 0.05}
    return pd.DataFrame(per_run), summary


def known_variant_sets(seed_dir, recs):
    """该种子、该方法的可见锚点集合：初始训练 + 此前轮次已揭示反馈。"""
    known = set(pd.read_csv(seed_dir / "train.csv").variant.astype(str))
    for row in recs:
        known.add(row["variant"])
    return known


def anchor_stats(global_fit, sources):
    """统计②：引用已测锚点的新候选实测分≥锚点实测分的比率，与随机基线比较。"""
    gf = global_fit.set_index("fitness").reset_index().set_index("variant")["fitness"]
    pair_rows = []
    for label, recs, seed_dir in sources:
        recs_sorted = sorted(recs, key=lambda r: (r["method"], int(r["round"]), int(r.get("rank", 0))))
        by_method = {}
        for row in recs_sorted:
            by_method.setdefault(row["method"], []).append(row)
        for method, rows in by_method.items():
            known = set(pd.read_csv(seed_dir / "train.csv").variant.astype(str))
            seen_rounds = {}
            for row in rows:
                text = f"{row.get('hypothesis') or ''} {row.get('rationale') or ''}"
                tokens = [t for t in AA4.findall(text.upper())
                          if t not in STOP and t != row["variant"] and t in known]
                anchors = list(dict.fromkeys(tokens))[:3]
                for anchor in anchors[:1] if len(anchors) == 1 else anchors:
                    if row["variant"] in gf.index and anchor in gf.index:
                        cand, anch = float(gf[row["variant"]]), float(gf[anchor])
                        base = float((global_fit.fitness >= anch).mean())
                        pair_rows.append({"source": label, "method": method, "round": row["round"],
                                          "variant": row["variant"], "anchor": anchor,
                                          "candidate_fitness": cand, "anchor_fitness": anch,
                                          "success": int(cand >= anch), "random_ge_anchor": base})
                known.add(row["variant"])
    pairs = pd.DataFrame(pair_rows)
    summary = {}
    if len(pairs):
        for (label, method), grp in pairs.groupby(["source", "method"]):
            summary[f"{label}:{method}"] = {
                "pairs": len(grp),
                "success_rate": round(grp.success.mean(), 4),
                "random_baseline": round(grp.random_ge_anchor.mean(), 4)}
        allgrp = pairs
        summary["ALL"] = {"pairs": len(allgrp), "success_rate": round(allgrp.success.mean(), 4),
                          "random_baseline": round(allgrp.random_ge_anchor.mean(), 4)}
    return pairs, summary


def main():
    global_fit = load_global_fitness()
    quantiles = {"top1pct_fitness": float(global_fit.fitness.quantile(0.99)),
                 "top5pct_fitness": float(global_fit.fitness.quantile(0.95)),
                 "global_top10": global_fit.nlargest(10, "fitness")[["variant", "fitness"]].to_dict("records")}
    print(json.dumps(quantiles, ensure_ascii=False, indent=2)[:800])

    runs = []
    for seed in (11, 22, 33):
        v1 = ROOT / "results" / f"seed-{seed}" / "run" / "events.jsonl"
        if v1.exists():
            rows = [r for r in first_answer_variants(v1) if r["method"] in ("llm", "knowledge")]
            runs.append((f"v1-seed{seed}", rows))
        v3 = ROOT / "results" / "prompt-v3" / f"seed-{seed}" / "run" / "events.jsonl"
        if v3.exists():
            rows = [r for r in first_answer_variants(v3) if r["method"] in ("llm", "knowledge")]
            runs.append((f"v3-seed{seed}", rows))
    per_run, overlap_summary = overlap_stats(global_fit, runs)
    per_run.to_csv(OUT / "first_answer_overlap.csv", index=False, encoding="utf-8-sig")

    sources = []
    for label, path, seed_dir in [
        ("v1", ROOT / "reports" / "all_recommendations.csv", None),
        ("v3", ROOT / "reports" / "validation-stage" / "v3_recommendations.csv", None),
    ]:
        recs = pd.read_csv(path).to_dict("records")
        if label == "v1":
            groups = []
            for seed in (11, 22, 33):
                sub = [r for r in recs if int(r["seed"]) == seed]
                groups.append((f"v1-seed{seed}", sub, ROOT / "data" / "processed" / f"seed-{seed}"))
            sources.extend(groups)
        else:
            for seed in (11, 22, 33):
                sub = [r for r in recs if int(r["seed"]) == seed]
                p = ROOT / "results" / "prompt-v3" / f"seed-{seed}" / "run" / "recommendations.csv"
                if p.exists():
                    sub = pd.read_csv(p).to_dict("records")
                sources.append((f"v3-seed{seed}", sub, ROOT / "data" / "processed" / f"seed-{seed}"))
    pairs, anchor_summary = anchor_stats(global_fit, sources)
    pairs.to_csv(OUT / "anchor_transfer.csv", index=False, encoding="utf-8-sig")

    summary = {"s1_quantiles": quantiles, "first_answer_overlap": overlap_summary,
               "anchor_transfer": anchor_summary}
    (OUT / "stats_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overlap": overlap_summary, "anchor": anchor_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
