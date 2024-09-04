import math
import os
from collections import defaultdict
from typing import Optional, Union

import modin.pandas as pd
import numpy as np
from alibaba_privacy_workload.utils import (
    compute_gaussian_demands,
    compute_laplace_demands,
    compute_noise_and_rdp_from_target_epsilon,
    map_to_range,
    sample_from_frequencies_dict,
)
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray.util.multiprocessing import Pool


class PrivacyWorkload:
    """
    csv-based privacy workload.

    Reads and processes the alibaba csv files; exposes features to be used as proxies and calls a deterministic
    function for the creation of the privacy alibaba workload
    """

    def __init__(self, alibaba_trace, cfg: DictConfig):
        self.cfg = cfg
        self.tasks = None
        self.alibaba_trace = alibaba_trace

    def create_dp_task(self, task_trace: Union[dict, pd.series.Series]) -> dict:
        inst_num = task_trace["inst_num"]
        start_execution_time = task_trace["start_execution_time"]
        end_execution_time = task_trace["end_execution_time"]
        plan_cpu = task_trace["plan_cpu"]
        plan_mem = task_trace["plan_mem"]
        plan_gpu = task_trace["plan_gpu"]
        gpu_type = task_trace["gpu_type"]
        submit_time = task_trace["submit_time"]
        cpu_usage = task_trace["cpu_usage"]
        gpu_wrk_util = task_trace["gpu_wrk_util"]
        avg_mem = task_trace["avg_mem"]
        max_mem = task_trace["max_mem"]
        avg_gpu_wrk_mem = task_trace["avg_gpu_wrk_mem"]
        max_gpu_wrk_mem = task_trace["max_gpu_wrk_mem"]
        read = task_trace["read"]
        write = task_trace["write"]
        read_count = task_trace["read_count"]
        write_count = task_trace["write_count"]
        machine_cpu_iowait = task_trace["machine_cpu_iowait"]
        machine_cpu_kernel = task_trace["machine_cpu_kernel"]
        machine_cpu_usr = task_trace["machine_cpu_usr"]
        machine_gpu = task_trace["machine_gpu"]
        machine_load_1 = task_trace["machine_load_1"]
        machine_net_receive = task_trace["machine_net_receive"]
        machine_num_worker = task_trace["machine_num_worker"]
        machine_cpu = task_trace["machine_cpu"]
        gpu_type_spec = task_trace["gpu_type_spec"]
        workload = task_trace["workload"]
        cap_cpu = task_trace["cap_cpu"]
        cap_mem = task_trace["cap_mem"]
        cap_gpu = task_trace["cap_gpu"]
        wait_time = start_execution_time - submit_time
        runtime = task_trace["runtime"]
        relative_submit_time = task_trace["relative_submit_time"]
        alphas = self.cfg.alphas
        cpu_based_curve = task_trace["cpu_based_curve"]

        n_blocks = self.compute_num_blocks(cpu_based_curve, inst_num, runtime, read)
        rdp_epsilons, epsilon, delta, task_name = self.compute_budget(
            cpu_task=cpu_based_curve,
            workload=workload,
            runtime=runtime,
            avg_mem=avg_mem,
            plan_gpu=plan_gpu,
            alphas=alphas,
            n_blocks=n_blocks,
            avg_gpu_wrk_mem=avg_gpu_wrk_mem,
        )

        task = {
            "epsilon": epsilon,
            "delta": delta,
            "n_blocks": n_blocks,
            "profit": self.compute_profit(gpu_type_spec),
            "block_selection_policy": self.compute_block_selection_policy(),
            "task_name": task_name,
            "alphas": alphas,
            "rdp_epsilons": rdp_epsilons,
            "relative_submit_time": relative_submit_time,
            "submit_time": submit_time,
            "workload": workload,
        }

        post_processed_task = self.post_process_task(task)
        return post_processed_task

    def post_process_task(self, task: dict) -> Optional[dict]:
        """
        Returns an error string if the task is too small or unsuitable for any other reason.
        Also clips the demands to avoid NaN.
        """

        task["rdp_epsilons"] = [
            1_000_000 if x == math.inf else x for x in task["rdp_epsilons"]
        ]

        # We normalize by a fictional block
        epsilon, delta = float(self.cfg.normalizing_epsilon), float(
            self.cfg.normalizing_delta
        )
        task["normalized_rdp_epsilons"] = []
        for alpha, rdp_epsilon in zip(task["alphas"], task["rdp_epsilons"]):
            block_epsilon = epsilon + np.log(delta) / (alpha - 1)
            if block_epsilon <= 0 or (rdp_epsilon / block_epsilon) == np.inf:
                task["normalized_rdp_epsilons"].append(10)
            else:
                task["normalized_rdp_epsilons"].append(rdp_epsilon / block_epsilon)

        epsilon_min = min(task["normalized_rdp_epsilons"])
        if epsilon_min < self.cfg.rdp_epsilon_min:
            return "epsilon_min_too_small"
        if epsilon_min > 1:
            return "epsilon_min_too_big"
        if task["n_blocks"] > self.cfg.n_blocks_cutoff:
            return "n_blocks_too_big"
        return task

    def generate(self):
        logger.info("Mapping to a privacy workload...")
        tasks_info = self.alibaba_trace.tasks_info.head(self.cfg.max_number_of_tasks)
        logger.info(tasks_info.head())

        logger.info(f"Enumerating tasks...")
        tasks_list = [index_and_row[1] for index_and_row in tasks_info.iterrows()]

        logger.info(f"Starting pool mapping on {len(tasks_list)} tasks...")
        with Pool(processes=int(os.cpu_count() * 0.75)) as pool:
            dp_tasks = list(pool.map(self.create_dp_task, tasks_list))

        valid_dp_tasks = []
        invalid_tasks = defaultdict(int)
        for t in dp_tasks:
            if isinstance(t, dict):
                valid_dp_tasks.append(t)
            else:
                invalid_tasks[t] += 1

        logger.info(f"Removed invalid tasks out of {len(dp_tasks)}: {invalid_tasks}")

        logger.info(f"Collecting results in a dataframe...")
        self.tasks = pd.DataFrame(valid_dp_tasks)

        logger.info(self.tasks.head())

    def dump(
        self,
        path,
    ):
        logger.info("Saving the privacy workload...")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks.to_csv(path, index=False)

        logger.info(f"Saved {len(self.tasks)} tasks at {path}.")
        OmegaConf.save(self.cfg, path.with_suffix(".cfg.yaml"))

    def compute_budget(
        self,
        cpu_task: bool,
        workload,
        runtime,
        avg_mem,
        plan_gpu,
        n_blocks,
        alphas,
        avg_gpu_wrk_mem,
    ):

        # Rough range of delta. The final demands are in RDP.
        dataset_size = n_blocks * self.cfg.avg_block_size
        delta = 1 / dataset_size

        if cpu_task == "True":
            # Epsilon ~ RAM * runtime
            epsilon = map_to_range(
                value=avg_mem * runtime,
                min_input=self.alibaba_trace.min_cpu_based_avg_mem_times_runtime,
                max_input=self.alibaba_trace.max_cpu_based_avg_mem_times_runtime,
                min_output=self.cfg.epsilon_min_cpu,
                max_output=self.cfg.epsilon_max,
            )

            # Number of epochs ~ runtime
            epochs = int(
                map_to_range(
                    value=runtime,
                    min_input=self.alibaba_trace.min_cpu_based_runtime,
                    max_input=self.alibaba_trace.max_cpu_based_runtime,
                    min_output=self.cfg.epochs_min,
                    max_output=self.cfg.epochs_max,
                )
            )

            # Map epsilon to an RDP curve with different mechanisms
            mechanism = sample_from_frequencies_dict(
                dict(self.cfg.cpu_mechanisms_frequencies)
            )
            params = {"epsilon": epsilon}
            if mechanism == "laplace":
                # Only one step for Laplace because they are already very costly RDP-wise
                rdp_epsilons = compute_laplace_demands(
                    laplace_noise=1 / epsilon, alphas=alphas
                )
            elif mechanism == "gaussian":
                rdp_epsilons = compute_gaussian_demands(
                    epsilon=epsilon, delta=delta, steps=epochs
                )
                params["delta"] = delta
                params["epochs"] = epochs

            elif mechanism == "subsampled_laplace":
                batch_size = int(
                    map_to_range(
                        value=plan_gpu,
                        min_input=self.alibaba_trace.min_cpu_based_inst_num,
                        max_input=self.alibaba_trace.max_cpu_based_inst_num,
                        min_output=self.cfg.batch_size_min,
                        max_output=min(self.cfg.batch_size_max, dataset_size // 10),
                    )
                )
                _, rdp_epsilons = compute_noise_and_rdp_from_target_epsilon(
                    target_epsilon=epsilon,
                    target_delta=delta,
                    dataset_size=dataset_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    alphas=alphas,
                    approx_ratio=0.1,
                    gaussian=False,
                )
                params["delta"] = delta
                params["epochs"] = epochs
                params["batch_size"] = batch_size

        else:
            # Epsilon ~ GPU mem * runtime
            epsilon = map_to_range(
                value=avg_gpu_wrk_mem * runtime,
                min_input=self.alibaba_trace.min_gpu_based_avg_gpu_wrk_mem_times_runtime,
                max_input=self.alibaba_trace.max_gpu_based_avg_gpu_wrk_mem_times_runtime,
                min_output=self.cfg.epsilon_min_gpu,
                max_output=self.cfg.epsilon_max,
            )

            # Batch size ~ GPU mem
            batch_size = int(
                map_to_range(
                    value=plan_gpu,
                    min_input=self.alibaba_trace.min_gpu_based_plan_gpu,
                    max_input=self.alibaba_trace.max_gpu_based_plan_gpu,
                    min_output=self.cfg.batch_size_min,
                    max_output=min(self.cfg.batch_size_max, dataset_size // 10),
                )
            )

            # Epochs ~ runtime
            epochs = int(
                map_to_range(
                    value=runtime,
                    min_input=self.alibaba_trace.min_gpu_based_runtime,
                    max_input=self.alibaba_trace.max_gpu_based_runtime,
                    min_output=self.cfg.epochs_min,
                    max_output=self.cfg.epochs_max,
                )
            )

            params = {"epsilon": epsilon}
            params["delta"] = delta

            mechanism = sample_from_frequencies_dict(
                dict(self.cfg.gpu_mechanisms_frequencies)
            )

            if mechanism == "subsampled_gaussian":
                sigma, rdp_epsilons = compute_noise_and_rdp_from_target_epsilon(
                    target_epsilon=epsilon,
                    target_delta=delta,
                    dataset_size=dataset_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    alphas=alphas,
                    approx_ratio=0.1,
                    gaussian=True,
                )
                params["epochs"] = epochs
                params["batch_size"] = batch_size

            elif mechanism == "dp_ftrl":
                rdp_epsilons = compute_gaussian_demands(
                    epsilon=epsilon,
                    delta=delta,
                    steps=int(np.ceil(np.log(dataset_size))),
                )

        param_string = ",".join([f"{k}={v:.2e}" for k, v in params.items()])
        task_name = f"{mechanism}-{param_string}"
        return rdp_epsilons, epsilon, delta, task_name

    def compute_num_blocks(self, cpu_task, inst_num, runtime, read):
        if cpu_task == "True":
            min_input = self.alibaba_trace.min_cpu_based_read
            max_input = self.alibaba_trace.max_cpu_based_read
            n_blocks = int(
                map_to_range(
                    value=read,
                    min_input=min_input,
                    max_input=max_input,
                    min_output=self.cfg.n_blocks_min,
                    max_output=self.cfg.n_blocks_max,
                )
            )
        else:
            min_input = self.alibaba_trace.min_gpu_based_read
            max_input = self.alibaba_trace.max_gpu_based_read
            n_blocks = int(
                map_to_range(
                    value=read,
                    min_input=min_input,
                    max_input=max_input,
                    min_output=self.cfg.n_blocks_min,
                    max_output=self.cfg.n_blocks_max,
                )
            )
        if n_blocks == 0:
            n_blocks = 1
        return n_blocks

    def compute_profit(self, gpu_type_spec):
        return 1

    def compute_block_selection_policy(self):
        return "LatestBlocksFirst"
