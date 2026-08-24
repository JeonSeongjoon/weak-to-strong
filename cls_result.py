"""
sample_difficulty/test_ds 결과 비교

입력 구조
    {BASE}/seed={seed}/test_ds/wms:{weak}_ms:{strong}/loss={loss}/*.json

비교 방식 — 모두 `대상 loss - 기준값`
    vs_xent      대상 loss  -  xent          (같은 관점 파일끼리)
    vs_weak      대상 loss  -  weak teacher
    vs_ceiling   대상 loss  -  strong ceiling
                 ceiling 은 self-pair(wms:{ms}_ms:{ms})의 weak_teacher_diff_info.json

출력
    per_seed.json    시드별 원시값
    aggregate.json   시드 across 평균·분산
    report.txt       사람이 읽는 정렬된 표
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ═══════════════════════════════════════════════════════════════ 설정

BASE = Path("./weak-to-strong/results/sample_difficulty")
OUT_DIR = Path("./weak-to-strong/results/comparison")

METRICS = ["acc_easy", "acc_overlap", "acc_hard", "acc_total"]

#  관점 이름  ->  파일 이름
PERSPECTIVES = {
    "student_own": "strong_student_result_info.json",
    "wk_perspec": "strong_student_result_in_wk_perspec.json",
    "ceil_perspec": "strong_student_result_in_ceil_perspec.json",
}

TARGET_LOSSES = ["conf_induc", "conf_induc_filt", "xent"]
BASELINE_LOSS = "xent"
WEAK_FILE = "weak_teacher_diff_info.json"

COMPARISONS = ["vs_xent", "vs_weak", "vs_ceiling"]

_PAIR_RE = re.compile(r"^wms:(?P<wms>.+)_ms:(?P<ms>.+)$")


# ═══════════════════════════════════════════════════════════════ 자료구조


@dataclass
class Scan:
    """디스크에서 읽어들인 원본 수치."""

    # (seed, wms, ms, loss) -> {perspective: {metric: value}}
    student: dict = field(default_factory=dict)
    # (seed, wms, ms) -> {metric: value}
    weak: dict = field(default_factory=dict)

    def ceiling(self, seed: int, ms: str) -> dict | None:
        """ceiling = 강한 모델이 자기 자신을 가르치는 쌍의 teacher."""
        return self.weak.get((seed, ms, ms))


# ═══════════════════════════════════════════════════════════════ 1. 수집


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _pick(d: dict) -> dict:
    return {m: d.get(m) for m in METRICS}


def scan_results() -> Scan:
    out = Scan()

    for seed_dir in sorted(BASE.glob("seed=*")):
        seed = int(seed_dir.name.split("=")[1])
        test_ds = seed_dir / "test_ds"
        if not test_ds.is_dir():
            continue

        for pair_dir in sorted(p for p in test_ds.iterdir() if p.is_dir()):
            matched = _PAIR_RE.match(pair_dir.name)
            if not matched:
                print(f"  [skip] 폴더명 형식 불일치 — {pair_dir.name}")
                continue
            wms, ms = matched["wms"], matched["ms"]

            for loss_dir in sorted(pair_dir.glob("loss=*")):
                loss = loss_dir.name.split("=", 1)[1]

                # weak teacher 정보는 loss 와 무관 → 처음 것만 취함
                if (seed, wms, ms) not in out.weak:
                    if (d := _read(loss_dir / WEAK_FILE)) is not None:
                        out.weak[(seed, wms, ms)] = _pick(d)

                views = {
                    name: _pick(d)
                    for name, fname in PERSPECTIVES.items()
                    if (d := _read(loss_dir / fname)) is not None
                }
                if views:
                    out.student[(seed, wms, ms, loss)] = views

    return out


# ═══════════════════════════════════════════════════════════════ 2. 비교


def compute_diffs(scan: Scan) -> list[dict]:
    """시드 × 쌍 × loss × 관점 × metric 마다 세 비교값을 담은 레코드."""
    records = []

    for (seed, wms, ms, loss), views in sorted(scan.student.items()):
        if loss not in TARGET_LOSSES:
            continue

        pair = f"{wms} -> {ms}"
        baseline = scan.student.get((seed, wms, ms, BASELINE_LOSS))
        weak_ref = scan.weak.get((seed, wms, ms))
        ceil_ref = scan.ceiling(seed, ms)

        for label, ref in (("xent", baseline), ("weak", weak_ref), ("ceiling", ceil_ref)):
            if ref is None:
                print(f"  [warn] {label} 기준값 없음 — seed={seed} {pair}")

        for view_name, values in views.items():
            for metric in METRICS:
                target = values.get(metric)
                if target is None:
                    continue

                refs = {
                    "vs_xent": baseline.get(view_name, {}).get(metric) if baseline else None,
                    "vs_weak": weak_ref.get(metric) if weak_ref else None,
                    "vs_ceiling": ceil_ref.get(metric) if ceil_ref else None,
                }

                rec = dict(
                    seed=seed,
                    pair=pair,
                    weak_model=wms,
                    strong_model=ms,
                    loss=loss,
                    perspective=view_name,
                    metric=metric,
                    target_value=round(target, 4),
                )
                for name, ref in refs.items():
                    rec[name] = (
                        None
                        if ref is None
                        else {"ref": round(ref, 4), "diff": round(target - ref, 4)}
                    )
                records.append(rec)

    return records


# ═══════════════════════════════════════════════════════════════ 3. 집계


def _stats(values: list[float], seeds: list[int]) -> dict:
    return {
        "n_seeds": len(values),
        "seeds": seeds,
        "mean": round(statistics.fmean(values), 4),
        "var": round(statistics.variance(values), 6) if len(values) > 1 else None,
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def aggregate(records: list[dict]) -> list[dict]:
    """시드를 가로질러 평균·분산. 모델쌍별로 유지."""
    grouped = defaultdict(list)
    for r in records:
        key = (r["pair"], r["weak_model"], r["strong_model"],
               r["loss"], r["perspective"], r["metric"])
        grouped[key].append(r)

    out = []
    for (pair, wms, ms, loss, view, metric), items in sorted(grouped.items()):
        row = dict(
            pair=pair,
            weak_model=wms,
            strong_model=ms,
            loss=loss,
            perspective=view,
            metric=metric,
        )
        for cmp_name in COMPARISONS:
            picked = [(r["seed"], r[cmp_name]["diff"])
                      for r in items if r.get(cmp_name) is not None]
            if not picked:
                row[cmp_name] = None
                continue
            picked.sort()
            row[cmp_name] = _stats([d for _, d in picked], [s for s, _ in picked])
        out.append(row)

    return out


# ═══════════════════════════════════════════════════════════════ 4. 리포트


def _fmt(cell: dict | None) -> str:
    """`+0.0123 ±0.004` 형태. 시드 1개면 편차 자리를 비움."""
    if cell is None:
        return f"{'—':>17}"
    mean = f"{cell['mean']:+.4f}"
    dev = f" ±{cell['std']:.3f}" if cell["std"] is not None else ""
    return f"{mean + dev:>17}"


def write_report(rows: list[dict], path: Path) -> None:
    lines: list[str] = []
    add = lines.append
    width = 96

    add("=" * width)
    add("난이도 그룹별 정확도 비교   (값 = 대상 loss − 기준,  양수면 대상 loss 가 우세)")
    add("시드 across 평균 ± 표준편차")
    add("=" * width)

    by_section = defaultdict(list)
    for r in rows:
        by_section[(r["perspective"], r["loss"])].append(r)

    for view, loss in sorted(by_section):
        add("")
        add(f"[ 관점: {view}    loss: {loss} ]")
        add("-" * width)
        add(f"{'모델 쌍':<26}{'metric':<14}" + "".join(f"{c:>17}" for c in COMPARISONS))
        add("-" * width)

        current_pair = None
        for r in sorted(by_section[(view, loss)],
                        key=lambda x: (x["pair"], METRICS.index(x["metric"]))):
            if r["pair"] != current_pair:
                if current_pair is not None:
                    add("")
                current_pair = r["pair"]
                shown = r["pair"]
            else:
                shown = ""
            add(f"{shown:<26}{r['metric']:<14}"
                + "".join(_fmt(r[c]) for c in COMPARISONS))

    add("")
    add("=" * width)
    add("주의: vs_weak 는 wk_perspec 관점에서, vs_ceiling 은 ceil_perspec 관점에서")
    add("      분류 기준이 같은 샘플 그룹끼리의 비교입니다.")
    add("      그 외 조합은 서로 다른 그룹의 정확도를 빼는 것이므로 해석에 주의하세요.")
    add("=" * width)

    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════ main


def main() -> None:
    print(f"스캔: {BASE.resolve()}")
    scan = scan_results()
    print(f"  student {len(scan.student)}건, weak {len(scan.weak)}건")

    if not scan.student:
        raise SystemExit("결과 없음 — BASE 경로를 확인하세요.")

    records = compute_diffs(scan)
    if not records:
        raise SystemExit("비교 가능한 조합 없음.")

    rows = aggregate(records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_seed.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "aggregate.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n저장: {OUT_DIR.resolve()}")
    print(f"  per_seed.json   {len(records)}건")
    print(f"  aggregate.json  {len(rows)}건\n")

    write_report(rows, OUT_DIR / "report.txt")


if __name__ == "__main__":
    main()