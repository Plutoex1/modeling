"""График значений, автокорреляция, гистограмма и подбор закона распределения."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moments import read_sequence, DATA_FILE

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAX_LAG = 10


def acf(x, lag):
    """Коэффициент автокорреляции со сдвигом lag (своя реализация формулы,
    без pandas): r(k) = cov(x_i, x_{i+k}) / var(x)."""
    centered = x - x.mean()
    num = np.dot(centered[:-lag], centered[lag:])
    den = np.dot(centered, centered)
    return num / den


def fit_hyperexponential(mean, cv):
    """Подбор параметров гиперэкспоненциального закона H2 по двум начальным
    моментам методом сбалансированных средних (p1/mu1 == p2/mu2 == mean/2)."""
    c2 = cv ** 2
    p1 = 0.5 * (1 + ((c2 - 1) / (c2 + 1)) ** 0.5)
    p2 = 1 - p1
    mu1 = 2 * p1 / mean
    mu2 = 2 * p2 / mean
    return p1, p2, mu1, mu2


def main():
    x = read_sequence(DATA_FILE)
    n = len(x)

    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, n + 1), x, color="steelblue", linewidth=0.9)
    plt.axhline(x.mean(), color="crimson", linestyle=":")
    plt.title("Значения заданной числовой последовательности")
    plt.xlabel("№ измерения")
    plt.ylabel("X")
    plt.tight_layout()
    plt.savefig(OUT / "01_values.png", dpi=200)
    plt.close()

    lags = list(range(1, MAX_LAG + 1))
    r = [acf(x, k) for k in lags]
    bound = 1.96 / n ** 0.5

    print("Коэффициенты автокорреляции заданной ЧП (сдвиг: значение)")
    for k, v in zip(lags, r):
        mark = " *" if abs(v) > bound else ""
        print(f"  {k:>2}: {v:+.4f}{mark}")
    print(f"граница случайности: ±{bound:.4f}\n")

    plt.figure(figsize=(7, 4))
    plt.bar(lags, r, color="steelblue")
    plt.axhline(bound, color="crimson", linestyle="--")
    plt.axhline(-bound, color="crimson", linestyle="--")
    plt.title("Автокорреляционная функция заданной ЧП")
    plt.xlabel("сдвиг k")
    plt.ylabel("r(k)")
    plt.tight_layout()
    plt.savefig(OUT / "02_acf.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(x, bins=20, color="darkorange", edgecolor="black", alpha=0.8, density=True)
    plt.title("Гистограмма распределения частот")
    plt.xlabel("X")
    plt.ylabel("плотность")
    plt.tight_layout()
    plt.savefig(OUT / "03_histogram.png", dpi=200)
    plt.close()

    mean, cv = x.mean(), x.std(ddof=1) / x.mean()
    print(f"mean = {mean:.4f}, cv = {cv:.4f}")

    if cv > 1:
        p1, p2, mu1, mu2 = fit_hyperexponential(mean, cv)
        print("cv > 1  =>  гиперэкспоненциальный закон H2")
        print(f"p1={p1:.4f}  p2={p2:.4f}  mu1={mu1:.5f}  mu2={mu2:.5f}")
        print(f"(средние фаз: 1/mu1={1/mu1:.3f}, 1/mu2={1/mu2:.3f})")
    else:
        print("cv <= 1: см. методику для др. закона (не требуется для этого варианта)")


if __name__ == "__main__":
    main()