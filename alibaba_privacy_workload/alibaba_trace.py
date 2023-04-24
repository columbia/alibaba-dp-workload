from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alibaba_privacy_workload.utils import get_df


class AlibabaTrace:
    def __init__(self, path, cache=True, n_days=30):
        path = Path(path)

        try:
            # Try to load from cache first
            if cache and path.exists() and path.is_file():
                logger.info("Loading from cache...")
                self.tasks_info = pd.read_csv(path)

            # Otherwise, load from the Alibaba directory
            else:
                if not path.exists() or path.is_file(e):
                    path = path.parent
                self.tasks_info = self.load_from_alibaba(path, n_days)

                # Cache precomputed dataframe for later
                if cache:
                    logger.info("Saving unmodified  tasks info to cache...")
                    self.tasks_info.to_csv(
                        path.joinpath(f"tasks_info_{n_days}_days.csv"), index=False
                    )
        except Exception as e:
            raise ValueError(
                f"path should be either the `data` dir of `cluster-trace-gpu-v2020` or a csv file. {e}"
            )

        self.precompute_metrics()

    def load_from_alibaba(self, path: Path, n_days: int) -> pd.DataFrame:
        logger.info("Loading Alibaba csv files..")
        dfj = get_df(path.joinpath("pai_job_table.csv"))
        dft = get_df(path.joinpath("pai_task_table.csv"))
        dfi = get_df(path.joinpath("pai_instance_table.csv"))
        dfs = get_df(path.joinpath("pai_sensor_table.csv"))
        dfg = get_df(path.joinpath("pai_group_tag_table.csv"))
        dfp = get_df(path.joinpath("pai_machine_spec.csv"))
        dfm = get_df(path.joinpath("pai_machine_metric.csv"))

        logger.info("Computing joins...")
        # Remove auxiliary information that will lead to duplicate columns
        # keep only the submit-time of jobs (we don't care for user-level scheduling at the moment)
        dfj = dfj.drop(columns={"end_time", "status", "user", "inst_id"})
        dfi = dfi.drop(columns=["start_time", "end_time", "status"])
        dfs = dfs.drop(
            columns=["job_name", "task_name", "inst_id", "machine", "gpu_name"]
        )
        dfm = dfm.drop(columns=["start_time", "end_time", "machine"])

        # We need groups for recurring tasks
        # dfg = dfg.drop(columns=["user", "group"])
        dfp = dfp.drop(columns=["gpu_type"])

        dfj.rename(columns={"start_time": "submit_time"}, inplace=True)
        dft = dft.rename(
            columns={
                "start_time": "start_execution_time",
                "end_time": "end_execution_time",
            },
            inplace=False,
        )

        # Primary Key of Task: <job_name,task_name>
        tasks = (
            dft.query('status == "Terminated"')
            .merge(dfj, on=["job_name"], how="inner")
            .sort_values("submit_time")
            .drop(columns=["status"])
        )
        instances_info = (
            dfi.merge(dfs, on=["worker_name"], how="inner")
            .merge(dfm, on=["worker_name"], how="inner")
            .merge(dfg, on=["inst_id"], how="inner")
        )

        instances_info = tasks.merge(
            instances_info.drop(columns=["inst_id"]),
            on=["job_name", "task_name"],
            how="inner",
        )
        instances_info = instances_info.merge(dfp, on=["machine"], how="inner").drop(
            columns=["machine", "worker_name", "inst_name"]
        )

        instances_info = (
            instances_info.groupby(["job_name", "task_name"])
            .agg(
                {
                    "inst_num": "first",
                    "start_execution_time": "first",
                    "end_execution_time": "first",
                    "plan_cpu": "first",
                    "plan_mem": "first",
                    "plan_gpu": "first",
                    "gpu_type": "first",
                    "submit_time": "first",
                    "cpu_usage": "mean",
                    "gpu_wrk_util": "mean",
                    "avg_mem": "mean",
                    "max_mem": "mean",
                    "avg_gpu_wrk_mem": "mean",
                    "max_gpu_wrk_mem": "mean",
                    "read": "sum",
                    "write": "sum",
                    "read_count": "sum",
                    "write_count": "sum",
                    "machine_cpu_iowait": "mean",
                    "machine_cpu_kernel": "mean",
                    "machine_cpu_usr": "mean",
                    "machine_gpu": "mean",
                    "machine_load_1": "mean",
                    "machine_net_receive": "mean",
                    "machine_num_worker": "mean",
                    "machine_cpu": "mean",
                    "gpu_type_spec": "first",
                    "workload": "first",
                    "cap_cpu": "first",
                    "cap_mem": "first",
                    "cap_gpu": "first",
                }
            )
            .reset_index()
        )

        logger.info(f"Number of (job,task): {len(instances_info)}")

        ii = (
            instances_info.groupby(["job_name"])
            .agg(
                {
                    "inst_num": "sum",
                    "start_execution_time": "mean",
                    "end_execution_time": "mean",
                    "plan_cpu": "first",
                    "plan_mem": "first",
                    "plan_gpu": "first",
                    "gpu_type": "first",
                    "submit_time": "first",
                    "cpu_usage": "mean",
                    "gpu_wrk_util": "mean",
                    "avg_mem": "mean",
                    "max_mem": "mean",
                    "avg_gpu_wrk_mem": "mean",
                    "max_gpu_wrk_mem": "mean",
                    "read": "sum",
                    "write": "sum",
                    "read_count": "sum",
                    "write_count": "sum",
                    "machine_cpu_iowait": "mean",
                    "machine_cpu_kernel": "mean",
                    "machine_cpu_usr": "mean",
                    "machine_gpu": "mean",
                    "machine_load_1": "mean",
                    "machine_net_receive": "sum",
                    "machine_num_worker": "mean",
                    "machine_cpu": "mean",
                    "gpu_type_spec": "first",
                    "workload": "first",
                    "cap_cpu": "first",
                    "cap_mem": "first",
                    "cap_gpu": "first",
                }
            )
            .reset_index()
        )

        # Keep tasks submitted over the first n_days
        ii["start_date"] = ii.submit_time.apply(
            pd.Timestamp, unit="s", tz="Asia/Shanghai"
        )
        min_start_date = pd.Timestamp("1970-03-04 06:26:42+08:00")
        max_start_date = min_start_date + pd.Timedelta(days=n_days)
        mask = (ii["start_date"] > min_start_date) & (
            ii["start_date"] <= max_start_date
        )  # bitwise `and` for Pandas series
        ii = ii.loc[mask]

        logger.info(f"Sticking to the first {n_days} days: {len(ii)} tasks.")
        ii = ii.dropna(
            subset=["read", "plan_mem", "plan_cpu", "avg_mem", "gpu_wrk_util"]
        )

        ii["runtime"] = ii["end_execution_time"] - ii["start_execution_time"]
        ii = ii.sort_values("submit_time")
        ii.drop(ii[ii["submit_time"] < 2500000.0].index, inplace=True)
        ii["relative_submit_time"] = (
            ii["submit_time"] - ii["submit_time"].shift(periods=1)
        ).fillna(0)

        logger.info(
            f"Finished loading the Alibaba tasks: {len(ii)} tasks (after dropping some NaN)."
        )

        return ii.reset_index()

    def precompute_metrics(self):
        """
        Compute some useful metrics from the task dataframe, and store them.
        """
        self.max_instances_times_runtime = (
            self.tasks_info["inst_num"] * self.tasks_info["runtime"]
        ).max()
        self.min_instances_times_runtime = (
            self.tasks_info["inst_num"] * self.tasks_info["runtime"]
        ).min()

        self.max_plan_mem_times_runtime = (
            self.tasks_info["plan_mem"] * self.tasks_info["runtime"]
        ).max()
        self.min_plan_mem_times_runtime = (
            self.tasks_info["plan_mem"] * self.tasks_info["runtime"]
        ).min()

        self.min_plan_gpu = self.tasks_info["plan_gpu"].min()
        self.max_plan_gpu = self.tasks_info["plan_gpu"].max()

        self.min_inst_num = self.tasks_info["inst_num"].min()
        self.max_inst_num = self.tasks_info["inst_num"].max()

        self.min_runtime = self.tasks_info["runtime"].min()
        self.max_runtime = self.tasks_info["runtime"].max()

        self.min_read = self.tasks_info["read"].min()
        self.max_read = self.tasks_info["read"].max()

        self.tasks_info["normalized_gpu_plan"] = (
            self.tasks_info["plan_gpu"] - self.min_plan_gpu
        ) / (self.max_plan_gpu - self.min_plan_gpu)
        self.tasks_info["cpu_based_curve"] = np.where(
            self.tasks_info["normalized_gpu_plan"] < 0.1, "True", "False"
        )
        cpu_based_curves = self.tasks_info.loc[
            self.tasks_info.cpu_based_curve == "True"
        ]
        gpu_based_curves = self.tasks_info.loc[
            self.tasks_info.cpu_based_curve == "False"
        ]

        self.min_gpu_based_plan_gpu = gpu_based_curves["plan_gpu"].min()
        self.max_gpu_based_plan_gpu = gpu_based_curves["plan_gpu"].max()

        self.min_cpu_based_read = cpu_based_curves["read"].min()
        self.max_cpu_based_read = cpu_based_curves["read"].max()
        self.min_gpu_based_read = gpu_based_curves["read"].min()
        self.max_gpu_based_read = gpu_based_curves["read"].max()

        self.min_cpu_based_runtime = cpu_based_curves["runtime"].min()
        self.max_cpu_based_runtime = cpu_based_curves["runtime"].max()
        self.min_gpu_based_runtime = gpu_based_curves["runtime"].min()
        self.max_gpu_based_runtime = gpu_based_curves["runtime"].max()

        self.min_cpu_based_inst_num = cpu_based_curves["inst_num"].min()
        self.max_cpu_based_inst_num = cpu_based_curves["inst_num"].max()
        self.min_gpu_based_inst_num = gpu_based_curves["inst_num"].min()
        self.max_gpu_based_inst_num = gpu_based_curves["inst_num"].max()

        self.min_cpu_based_read_times_inst_num = (
            cpu_based_curves["read"] * cpu_based_curves["inst_num"]
        ).min()
        self.max_cpu_based_read_times_inst_num = (
            cpu_based_curves["read"] * cpu_based_curves["inst_num"]
        ).max()
        self.min_gpu_based_read_times_inst_num = (
            gpu_based_curves["read"] * gpu_based_curves["inst_num"]
        ).min()
        self.max_gpu_based_read_times_inst_num = (
            gpu_based_curves["read"] * gpu_based_curves["inst_num"]
        ).max()

        self.min_cpu_based_avg_mem_times_runtime = (
            cpu_based_curves["avg_mem"] * cpu_based_curves["runtime"]
        ).min()
        self.max_cpu_based_avg_mem_times_runtime = (
            cpu_based_curves["avg_mem"] * cpu_based_curves["runtime"]
        ).max()

        self.min_gpu_based_avg_gpu_wrk_mem_times_runtime = (
            gpu_based_curves["avg_gpu_wrk_mem"] * gpu_based_curves["runtime"]
        ).min()
        self.max_gpu_based_avg_gpu_wrk_mem_times_runtime = (
            gpu_based_curves["avg_gpu_wrk_mem"] * gpu_based_curves["runtime"]
        ).max()
