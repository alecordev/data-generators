import os
import random
import logging

import pandas as pd
import numpy as np


logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] - %(message)s")
os.chdir(os.path.abspath(os.path.dirname(__file__)))
logging.info(os.getcwd())


def reservoir_sampling(iterable, n):
    """
    Returns @param n random items from @param iterable.
    """
    reservoir = []
    for t, item in enumerate(iterable):
        if t < n:
            reservoir.append(item)
        else:
            m = random.randint(0, t)
            if m < n:
                reservoir[m] = item
    return reservoir


def generate_data():
    categories = ["cat1", "cat2", "cat3", "cat4", "cat5"]
    np.random.seed(1)

    data = pd.DataFrame(
        {
            "date": pd.date_range(start="20150101", periods=1000),
            "importance": [random.choice(list(range(10))) for _ in range(1000)],
            "col1": np.random.randn(1000),
            "col2": np.random.randn(1000),
            "col3": np.random.randn(1000),
            "num": [np.random.randint(1, 10) for _ in range(1000)],
            "num2": [np.random.randint(10, 20) for _ in range(1000)],
            "initial": np.random.uniform(low=15000000, high=500000000, size=(1000,)),
            "final": np.random.uniform(low=10000000, high=400000000, size=(1000,)),
            "size": np.random.uniform(low=10e10, high=9e12, size=(1000,)),
            "dist": np.random.exponential(scale=1e10, size=(1000,)),
            "normal": np.random.normal(scale=1.0, size=(1000,)),
            "category": [random.choice(categories) for _ in range(1000)],
        }
    )

    data["num3"] = data["num"] * np.random.lognormal(mean=0.5, sigma=1.0)
    data["diff"] = data["final"] - data["initial"]
    return data


def generate_entity_data(runs=100):
    banks = ["entity{}".format(i) for i in range(12)]
    np.random.seed(1)

    runs_data = []

    for _ in range(runs):
        banks_in_run = reservoir_sampling(banks, 10)
        date = pd.date_range(start="20150101", periods=1000)[
            np.random.randint(0, 999)
        ].to_pydatetime()
        run_id = ["FXD" + date.strftime("%y%m%d") for _ in range(10)]
        total_saving = np.random.uniform(low=15000000, high=500000000)
        trades = np.random.randint(low=20, high=500, size=(10,))

        nums = np.random.uniform(size=(10,))
        total = np.sum(nums)
        vals = [(i / total) * total_saving for i in nums]

        run_data = pd.DataFrame(
            {
                "run_id": run_id,
                "entity": banks_in_run,
                "date": date,
                "total_saving": total_saving,
                "net_saving": vals,
                "trades": trades,
            }
        )
        runs_data.append(run_data)

    return pd.concat(runs_data)


if __name__ == "__main__":
    pass
