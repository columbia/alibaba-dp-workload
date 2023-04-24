# Alibaba-DP Workload


This repository contains Alibaba-DP, a differential privacy (DP) workload based on the [2020 Alibaba GPU Trace](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020).
The original trace contains resource demands and allocation for tasks running on a production cluster.
We use these metrics as proxies for task privacy resource demands, which do not exist in the original trace.

The Alibaba-DP workload can be used as a macrobenchmark to evaluate scheduling algorithms for differential privacy, such as DPF, from [PrivateKube / Privacy Budget Scheduling (OSDI'21)](https://arxiv.org/abs/2106.15335) or DPK, from [Packing Privacy Budget Efficiently](https://arxiv.org/abs/2212.13228). 
Since there are no publicly available real-world traces of DP workloads, previous privacy scheduling algorithms were evaluated on purely synthetic microbenchmarks. The Alibaba-DP trace is at least partially backed by real-world workload patterns.

We hope this trace can help other DP researchers evaluate their own algorithms. Our work can be cited as:

```bibtex
@misc{tholoniat2022packing,
      title={Packing Privacy Budget Efficiently}, 
      author={Pierre Tholoniat and Kelly Kostopoulou and Mosharaf Chowdhury and Asaf Cidon and Roxana Geambasu and Mathias Lécuyer and Junfeng Yang},
      year={2022},
      eprint={2212.13228},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

## Methodology

Our methodology is detailed in [Packing Privacy Budget Efficiently](https://arxiv.org/abs/2212.13228):

> We use machine type (CPU/GPU) as a proxy for DP mechanism type. We assume CPU-based tasks correspond to mechanisms used for statistics, analytics, or lightweight ML (e.g. XGBoost or decision trees), while GPU-based tasks correspond to deep learning mechanisms (DP-SGD or DP-FTRL). We map each CPU-based task to one of the {Laplace, Gaussian, Subsampled Laplace} curves and each GPU-based task to one of the {composition of Subsampled Gaussians, composition of Gaussians} curves, at random. We use memory usage as a proxy for privacy usage by setting traditional DP ε as an affine transformation of memory usage (in GB hours). We don’t claim that memory will be correlated with privacy in a realistic DP workload, but that the privacy budget might follow a similar distribution (e.g. a power law with many tasks having small requests and a long tail of tasks with large requests). We compute the number of blocks required by each task as an affine function of the bytes read through the network. Unlike the privacy budget proxy, we expect this proxy to have at least some degree of realism when data is stored remotely: tasks that don’t communicate much over the network are probably not using large portions of the dataset. Finally, all tasks request the most recent blocks that
arrived in the system and are assigned a weight of 1. We truncate the workload by sampling one month of the total trace and cutting off tasks that request more than 100 blocks or whose smallest normalized RDP ε is not in [0.001, 1]. The resulting workload, called Alibaba-DP, is an objectively derived version of the Alibaba trace.

The resulting trace is available as a `.csv` file here: [sample_output/privacy_tasks_30_days.csv](sample_output/privacy_tasks_30_days.csv). The DP budget is expressed using Rényi DP, with normalized demands added for convenience.

## Generating your own trace

We invite researchers to produce their own trace by modifying our [default constants](alibaba_privacy_workload/config/_default.yaml) or using different proxies for the privacy resource in the [generation script](alibaba_privacy_workload/privacy_workload.py).


1. Clone this repository and `cd` into it:
 
```bash
git clone https://github.com/columbia/alibaba-dp-workload.git
cd alibaba-dp-workload
```

2. Create a new virtual environment with the dependencies, e.g. with Poetry:
```bash
# Install Poetry if necessary
curl -sSL https://install.python-poetry.org | python3 -

# Activate the environment and install the dependencies
poetry shell 
poetry install
```

3. Download the original GPU trace from Alibaba:

```bash
bash download_alibaba_data.sh
```

4. Generate a workload with `python alibaba_privacy_workload/generate.py`. 

It takes about 15-45 mins on 32 cores. 
The output is a DP workload stored in `outputs/privacy_tasks_{cfg.n_days}_days.csv`.

Example usage:
```bash
python alibaba_privacy_workload/generate.py max_number_of_tasks=1000

python alibaba_privacy_workload/generate.py profits.default=0.5 epsilon_min=0.1

python alibaba_privacy_workload/generate.py --help
```
