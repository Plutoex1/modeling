"""Генерация модельной ЧП по закону H2 и сравнение с заданной последовательностью."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moments import read_sequence, estimate, deviation_pct, SAMPLE_SIZES, DATA_FILE
from sequence_analysis import acf, fit_hyperexponential, OUT, MAX_LAG

# параметры H2, полученные в sequence_analysis.py (метод сбалансированных средних)
P1, P2, MU1, MU2 = 0.9274, 0.0726, 0.19817, 0.01551


def hyperexp_sample(size, p1, mu1, mu2, generator):
    phase = generator.random(size) < p1
    u = generator.random(size)
    return np.where(phase, -np.log(u) / mu1, -np.log(u) / mu2)


def compare_tables(target, model):
    target_stats = {n: estimate(target[:n]) for n in SAMPLE_SIZES}
    model_stats = {n: estimate(model[:n]) for n in SAMPLE_SIZES}

    col_w = 11
    head = "n".ljust(18) + "".join(str(n).rjust(col_w) for n in SAMPLE_SIZES)
    print("ФОРМА 2 — характеристики сгенерированной ЧП (закон H2)")
    print(head)
    print("=" * len(head))

    fields = [("Мат. ожидание", "mean"), ("Дисперсия", "disp"), ("СКО", "std"), ("Коэф. вариации", "cv")]
    for label, key in fields:
        vals = [model_stats[n][key] for n in SAMPLE_SIZES]
        refs = [target_stats[n][key] for n in SAMPLE_SIZES]
        print(label.ljust(18) + "".join(f"{v:>{col_w}.4f}" for v in vals))
        print("  откл.,%".ljust(18) + "".join(
            f"{deviation_pct(v, r):>{col_w}.2f}" for v, r in zip(vals, refs)))
        print("-" * len(head))

    return target_stats, model_stats


def compare_autocorrelation(target, model):
    lags = list(range(1, MAX_LAG + 1))
    r_target = [acf(target, k) for k in lags]
    r_model = [acf(model, k) for k in lags]

    print("\nФОРМА 3 — автокорреляция: заданная / сгенерированная")
    print("сдвиг " + "".join(f"{k:>8}" for k in lags))
    print("задан." + "".join(f"{v:>8.4f}" for v in r_target))
    print("модель" + "".join(f"{v:>8.4f}" for v in r_model))

    plt.figure(figsize=(7.5, 4))
    width = 0.35
    idx = np.array(lags)
    plt.bar(idx - width / 2, r_target, width=width, label="заданная", color="steelblue")
    plt.bar(idx + width / 2, r_model, width=width, label="сгенерир.", color="darkorange")
    plt.axhline(0, color="black", linewidth=0.7)
    plt.legend()
    plt.xlabel("сдвиг")
    plt.ylabel("r(k)")
    plt.title("Автокорреляция: заданная vs сгенерированная")
    plt.tight_layout()
    plt.savefig(OUT / "04_acf_compare.png", dpi=200)
    plt.close()


def main():
    target = read_sequence(DATA_FILE)
    rng = np.random.default_rng(seed=1917)
    model = hyperexp_sample(len(target), P1, MU1, MU2, rng)

    compare_tables(target, model)
    compare_autocorrelation(target, model)

    corr = np.corrcoef(target, model)[0, 1]
    print(f"\nКоэффициент корреляции (заданная / сгенерированная): {corr:.4f}")

    plt.figure(figsize=(9, 4.5))
    plt.plot(target, color="steelblue", linewidth=0.8, label="заданная")
    plt.plot(model, color="darkorange", linewidth=0.8, alpha=0.7, label="сгенерированная")
    plt.legend()
    plt.title("Сравнение значений последовательностей")
    plt.tight_layout()
    plt.savefig(OUT / "05_values_compare.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(target, bins=20, alpha=0.6, density=True, label="заданная", color="steelblue")
    plt.hist(model, bins=20, alpha=0.6, density=True, label="сгенерированная", color="darkorange")
    plt.xlim(0, 100)
    plt.legend()
    plt.title("Сравнение гистограмм")
    plt.tight_layout()
    plt.savefig(OUT / "06_hist_compare.png", dpi=200)
    plt.close()

    np.savetxt(OUT / "generated.txt", model, fmt="%.5f")
    print(f"\nМодельная последовательность сохранена: {OUT / 'generated.txt'}")


if __name__ == "__main__":
    main()