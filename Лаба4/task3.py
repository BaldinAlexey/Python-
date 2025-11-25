import numpy as np
import pandas as pd
import polars as pl
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

def generate_large_dataset(n_rows=1_000_000):
    return pd.DataFrame({
        'id': range(n_rows),
        'timestamp': pd.date_range('2020-01-01', periods=n_rows, freq='1min'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'value1': np.random.normal(0, 1, n_rows),
        'value2': np.random.exponential(1, n_rows),
        'value3': np.random.randint(0, 100, n_rows)
    })


def benchmark(func, *args, **kwargs):
    start_mem = psutil.Process().memory_info().rss / 1024 ** 2
    start = time.time()

    result = func(*args, **kwargs)

    end = time.time()
    end_mem = psutil.Process().memory_info().rss / 1024 ** 2

    return {
        "Результат": result,
        "Время": end - start,
        "Память": end_mem - start_mem
    }


def pandas_filter_agg(df):
    return df[df["value1"] > 0].groupby("category")["value2"].mean()

def pandas_arrow_filter_agg(df):
    df2 = df.astype({"category": "string[pyarrow]"})
    return df2[df2["value1"] > 0].groupby("category")["value2"].mean()

def polars_filter_agg(df):
    pl_df = pl.from_pandas(df)
    return (
        pl_df.filter(pl.col("value1") > 0)
             .group_by("category")
             .agg(pl.mean("value2"))
    )


def pandas_groupby(df):
    # Берём только числовые столбцы для агрегации
    numeric_cols = ["value1", "value2", "value3"]

    return (
        df.groupby("category")[numeric_cols]
        .agg(["sum", "mean", "count"])
    )


def pandas_arrow_groupby(df):
    df2 = df.astype({"category": "string"})
    numeric_cols = ["value1", "value2", "value3"]

    return (
        df2.groupby("category")[numeric_cols]
        .agg(["sum", "mean", "count"])
    )


def polars_groupby(df):
    pl_df = pl.from_pandas(df)

    return (
        pl_df
        .group_by("category")
        .agg([
            pl.col("value1").sum().alias("value1_sum"),
            pl.col("value1").mean().alias("value1_mean"),
            pl.col("value1").count().alias("value1_count"),

            pl.col("value2").sum().alias("value2_sum"),
            pl.col("value2").mean().alias("value2_mean"),
            pl.col("value2").count().alias("value2_count"),

            pl.col("value3").sum().alias("value3_sum"),
            pl.col("value3").mean().alias("value3_mean"),
            pl.col("value3").count().alias("value3_count"),
        ])
    )



def pandas_join(df):
    df2 = df.sample(50_000)
    return df.merge(df2, on="id", how="left")

def pandas_arrow_join(df):
    df2 = df.sample(50_000).astype({"category": "string[pyarrow]"})
    return df.astype({"category": "string[pyarrow]"}).merge(df2, on="id", how="left")

def polars_join(df):
    pl_df = pl.from_pandas(df)
    join_df = pl_df.sample(50_000)
    return pl_df.join(join_df, on="id", how="left")


def pandas_rolling(df):
    return df["value1"].rolling(100).mean()

def pandas_arrow_rolling(df):
    df2 = df.astype({"category": "string[pyarrow]"})
    return df2["value1"].rolling(100).mean()

def polars_rolling(df):
    pl_df = pl.from_pandas(df)
    return pl_df.select(pl.col("value1").rolling_mean(100))


def pandas_resample(df):
    return df.resample("1h", on="timestamp")["value1"].mean()

def pandas_arrow_resample(df):
    df2 = df.astype({"category": "string[pyarrow]"})
    return df2.resample("1h", on="timestamp")["value1"].mean()

def polars_resample(df):
    pl_df = pl.from_pandas(df)
    return (
        pl_df.group_by_dynamic("timestamp", every="1h")
             .agg(pl.mean("value1"))
    )


def run_all_benchmarks(df):
    operations = {
        "filter_agg": (pandas_filter_agg, pandas_arrow_filter_agg, polars_filter_agg),
        "groupby": (pandas_groupby, pandas_arrow_groupby, polars_groupby),
        "join": (pandas_join, pandas_arrow_join, polars_join),
        "rolling": (pandas_rolling, pandas_arrow_rolling, polars_rolling),
        "resample": (pandas_resample, pandas_arrow_resample, polars_resample),
    }

    results = {}

    for name, (p_fun, pa_fun, pl_fun) in operations.items():
        results[name] = {
            "pandas": benchmark(p_fun, df),
            "pandas+pyarrow": benchmark(pa_fun, df),
            "polars": benchmark(pl_fun, df)
        }

    return results


def generate_comparison_report(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Сравнение производительности Pandas, Pandas+PyArrow и Polars")

    # --- Heatmap времени ---
    time_data = pd.DataFrame({
        op: {
            impl: results[op][impl]["time"]
            for impl in results[op]
        }
        for op in results
    })
    sns.heatmap(time_data, annot=True, cmap="coolwarm", ax=axes[0, 0])
    axes[0, 0].set_title("Время выполнения")

    # --- Heatmap памяти ---
    mem_data = pd.DataFrame({
        op: {
            impl: results[op][impl]["memory"]
            for impl in results[op]
        }
        for op in results
    })
    sns.heatmap(mem_data, annot=True, cmap="Greens", ax=axes[0, 1])
    axes[0, 1].set_title("Пиковая память (MB)")

    plt.tight_layout()
    plt.show()


def main():
    print("Генерируем датасет...")
    df = generate_large_dataset(500_000)  # полмиллиона записей

    print("Запускаем бенчмарк...")
    results = run_all_benchmarks(df)

    print("Строим отчёт...")
    generate_comparison_report(results)


if __name__ == "__main__":
    main()

