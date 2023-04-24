from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from alibaba_privacy_workload.alibaba_trace import AlibabaTrace
from alibaba_privacy_workload.privacy_workload import PrivacyWorkload

WORKLOAD_REPO_ROOT = Path(__file__).resolve().parent.parent


@hydra.main(config_path="config", config_name="_default")
def main(cfg: DictConfig) -> None:
    logger.info(cfg)

    original_tasks_cache = WORKLOAD_REPO_ROOT.joinpath(
        f"cluster-trace-gpu-v2020/data/tasks_info_{cfg.n_days}_days.csv"
    )
    alibaba_trace = AlibabaTrace(
        original_tasks_cache, cache=cfg.cache, n_days=cfg.n_days
    )

    privacy_alibaba_workload = PrivacyWorkload(alibaba_trace, cfg)
    privacy_alibaba_workload.generate()

    privacy_tasks_path = WORKLOAD_REPO_ROOT.joinpath(
        f"outputs/privacy_tasks_{cfg.n_days}_days.csv"
    )
    privacy_alibaba_workload.dump(path=privacy_tasks_path)


if __name__ == "__main__":
    main()
