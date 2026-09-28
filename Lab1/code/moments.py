"""Форма 1: числовые моменты и доверительные интервалы заданной ЧП."""
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).parent / "var209.csv"
SAMPLE_SIZES = (10, 20, 50, 100, 200, 300)
T_COEFFS = {0.90: 1.643, 0.95: 1.960, 0.99: 2.576}


def read_sequence(path):
    text = path.read_text(encoding="utf-8")
    return np.array([float(row.replace(",", ".")) for row in text.splitlines() if row.strip()])


def estimate(sample):
    n = len(sample)
    mean = sample.mean()
    disp = sample.var(ddof=1)
    std = disp ** 0.5
    cv = std / mean
    sigma_mean = (disp / n) ** 0.5
    return {
        "n": n,
        "mean": mean,
        "disp": disp,
        "std": std,
        "cv": cv,
        "ci": {p: k * sigma_mean for p, k in T_COEFFS.items()},
    }


def deviation_pct(current, base):
    return (current - base) / base * 100.0


def report_table(sequence):
    stats = {n: estimate(sequence[:n]) for n in SAMPLE_SIZES}
    baseline = stats[300]

    rows = [
        ("Мат. ожидание", lambda s: s["mean"]),
        ("Дов.инт. 0.90", lambda s: s["ci"][0.90]),
        ("Дов.инт. 0.95", lambda s: s["ci"][0.95]),
        ("Дов.инт. 0.99", lambda s: s["ci"][0.99]),
        ("Дисперсия", lambda s: s["disp"]),
        ("СКО", lambda s: s["std"]),
        ("Коэф. вариации", lambda s: s["cv"]),
    ]

    col_w = 11
    head = "n".ljust(18) + "".join(str(n).rjust(col_w) for n in SAMPLE_SIZES)
    print("ФОРМА 1 — характеристики заданной ЧП (вариант 209)")
    print(head)
    print("=" * len(head))
    for label, getter in rows:
        vals = [getter(stats[n]) for n in SAMPLE_SIZES]
        base = getter(baseline)
        print(label.ljust(18) + "".join(f"{v:>{col_w}.4f}" for v in vals))
        print("  откл.,%".ljust(18) + "".join(f"{deviation_pct(v, base):>{col_w}.2f}" for v in vals))
        print("-" * len(head))

    return stats


if __name__ == "__main__":
    seq = read_sequence(DATA_FILE)
    report_table(seq)