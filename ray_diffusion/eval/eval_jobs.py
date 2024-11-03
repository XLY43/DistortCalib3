"""
python -m ray_diffusion.eval.eval_jobs --eval_type diffusion --eval_path models/co3d_diffusion
"""

import argparse
import itertools
import json
import os
from glob import glob

import numpy as np
from tqdm.auto import tqdm

from ray_diffusion.dataset.co3d_v2 import TEST_CATEGORIES, TRAINING_CATEGORIES
from ray_diffusion.eval.eval_category import save_results
import submitit
from submitit.helpers import Checkpointable, DelayedSubmission

import os
from enum import Enum
from typing import Optional

START, END = 6, 9

class SlurmJobType(Enum):
    CPU = 0
    GPU = 1

def setup_slurm(
    name: str,
    job_type: SlurmJobType,
    submitit_folder: str = "submitit",
    depend_on: Optional[str] = None,
    timeout: int = 180,
    high_compute_memory: bool = False,
) -> submitit.AutoExecutor:
    os.makedirs(submitit_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=submitit_folder, slurm_max_num_timeout=10)

    ################################################
    ##                                            ##
    ##   ADAPT THESE PARAMETERS TO YOUR CLUSTER   ##
    ##                                            ##
    ################################################

    # You may choose low-priority partitions where job preemption is enabled as
    # any preempted jobs will automatically resume/restart when rescheduled.

    if job_type == SlurmJobType.CPU:
        kwargs = {
            "slurm_partition": "compute",
            "gpus_per_node": 0,
            "slurm_cpus_per_task": 14,
            "slurm_mem": "32GB" if not high_compute_memory else "64GB",
        }
    elif job_type == SlurmJobType.GPU:
        kwargs = {
            "partition": "GPU-shared",
            # "gpus_per_node": 1,
            "gres": "gpu:v100-32:1",
            "nodes": 1,
            # "slurm_cpus_per_task": 1,
            # "slurm_mem": "16GB",
            # If your cluster supports choosing specific GPUs based on constraints,
            # you can uncomment this line to select low-memory GPUs.
            # "slurm_constraint": "p40",
        }

    ###################
    ##               ##
    ##   ALL DONE!   ##
    ##               ##
    ###################

    # kwargs = {
    #     **kwargs,
    #     "slurm_job_name": name,
    #     "timeout_min": timeout,
    #     "tasks_per_node": 1,
    #     # "slurm_additional_parameters": {"depend": f"afterany:{depend_on}"}
    #     # if depend_on is not None
    #     # else {},
    # }

    executor.update_parameters(**kwargs)
    executor.update_parameters(slurm_array_parallelism=1)
    # executor.update_parameters(tasks_per_node=1)
    return executor

class CategoryInference(Checkpointable):
    def __call__(self, **job_config):
        return save_results(**job_config)

    def checkpoint(self, **job_config) -> DelayedSubmission:
        """Resubmits the same callable with the same arguments"""
        return DelayedSubmission(self, **job_config)  # type: ignore

def is_slurm_available() -> bool:
    return submitit.AutoExecutor(".").cluster == "slurm"

def evaluate_ray_diffusion(eval_path):
    JOB_PARAMS = {
        "output_dir": [eval_path],
        "checkpoint": [450_000],
        # "num_images": [2, 3, 4, 5, 6, 7, 8],
        'num_images': [6, 7, 8],
        "category": TEST_CATEGORIES,
        # "category": ["apple"],
        "calculate_additional_timesteps": [True],
        "sample_num": [0, 1],
        "rescale_noise": ["zero"],  # Don't add noise during DDPM
        "normalize_moments": [True],
    }
    keys, values = zip(*JOB_PARAMS.items())
    job_configs = [dict(zip(keys, p)) for p in itertools.product(*values)]

    if is_slurm_available():
        print("SLURM is available")

        executor = setup_slurm(
                    f"distortcalib",
                    SlurmJobType.GPU,
                    timeout=24 * 60,
                )

        with executor.batch():
            for job in job_configs:
                # import pdb; pdb.set_trace()
                try:
                    executor.submit(CategoryInference(), job)
                except:
                    import pdb; pdb.set_trace()

        print(f"Submitted {len(job_configs)} jobs to SLURM")
    else:
        for job_config in tqdm(job_configs):
            # You may want to parallelize these jobs here, e.g. with submitit.
            save_results(**job_config)


# def evaluate_ray_regression(eval_path):
#     JOB_PARAMS = {
#         "output_dir": [eval_path],
#         "checkpoint": [300_000],
#         "num_images": [2, 3, 4, 5, 6, 7, 8],
#         # 'num_images': [2],
#         # "category": TRAINING_CATEGORIES + TEST_CATEGORIES,
#         "category": ["apple"],
#         "sample_num": [0, 1, 2, 3, 4],
#     }
#     keys, values = zip(*JOB_PARAMS.items())
#     job_configs = [dict(zip(keys, p)) for p in itertools.product(*values)]
#     for job_config in tqdm(job_configs[::-1]):
#         # You may want to parallelize these jobs here, e.g. with submitit.
#         save_results(**job_config)


def process_predictions(eval_path, pred_index):
    """
    pred_index should be 0 for regression and 7 for diffusion (corresponding to T=30)
    """
    errors = {
        c: {n: {"CC": [], "R": []} for n in range(START, END)}
        for c in TEST_CATEGORIES
    }

    for category in tqdm(TEST_CATEGORIES): # TRAINING_CATEGORIES + TEST_CATEGORIES
        for num_images in range(START, END):
            for sample_num in range(5):
                data_path = glob(
                    os.path.join(
                        eval_path,
                        "eval",
                        f"{category}_{num_images}_{sample_num}_ckpt*.json",
                    )
                )[0]
                with open(data_path) as f:
                    eval_data = json.load(f)

                for preds in eval_data.values():
                    errors[category][num_images]["R"].extend(
                        preds[pred_index]["R_error"]
                    )
                    errors[category][num_images]["CC"].extend(
                        preds[pred_index]["CC_error"]
                    )

    threshold_R = 15
    threshold_CC = 0.1

    all_seen_acc_R = []
    all_seen_acc_CC = []
    all_seen_acc_ray = []
    all_unseen_acc_R = []
    all_unseen_acc_CC = []
    all_unseen_acc_ray = []
    for num_images in range(START, END):
        seen_acc_R = []
        seen_acc_CC = []
        seen_acc_ray = []
        unseen_acc_R = []
        unseen_acc_CC = []
        unseen_acc_ray = []
        for category in TEST_CATEGORIES: #TEST_CATEGORIES
            unseen_acc_R.append(
                np.mean(np.array(errors[category][num_images]["R"]) < threshold_R)
            )
            unseen_acc_CC.append(
                np.mean(np.array(errors[category][num_images]["CC"]) < threshold_CC)
            )
            # unseen_acc_ray.append(
            #     np.mean(np.array(errors[category][num_images]["Rays_error"]))
            # )
        for category in TRAINING_CATEGORIES: #TRAINING_CATEGORIES
            seen_acc_R.append(
                np.mean(np.array(errors[category][num_images]["R"]) < threshold_R)
            )
            seen_acc_CC.append(
                np.mean(np.array(errors[category][num_images]["CC"]) < threshold_CC)
            )
            # seen_acc_ray.append(
            #     np.mean(np.array(errors[category][num_images]["Rays_error"]))
            # )
        all_seen_acc_R.append(np.mean(seen_acc_R))
        all_seen_acc_CC.append(np.mean(seen_acc_CC))
        all_unseen_acc_R.append(np.mean(unseen_acc_R))
        all_unseen_acc_CC.append(np.mean(unseen_acc_CC))
        # all_seen_acc_ray.append(np.mean(seen_acc_ray))
        # all_unseen_acc_ray.append(np.mean(unseen_acc_ray))

    print("N=       ", " ".join(f"{i: 5}" for i in range(START, END)))
    print("Seen R   ", " ".join([f"{x:0.3f}" for x in all_seen_acc_R]))
    print("Seen CC  ", " ".join([f"{x:0.3f}" for x in all_seen_acc_CC]))
    # print("Seen Ray ", " ".join([f"{x:0.3f}" for x in all_seen_acc_ray]))
    print("Unseen R ", " ".join([f"{x:0.3f}" for x in all_unseen_acc_R]))
    print("Unseen CC", " ".join([f"{x:0.3f}" for x in all_unseen_acc_CC]))
    # print("Unseen Ray", " ".join([f"{x:0.3f}" for x in all_unseen_acc_ray]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_type", type=str, default="diffusion", help="diffusion or regression"
    )
    parser.add_argument("--eval_path", type=str, default=None)
    args = parser.parse_args()

    eval_type = args.eval_type
    eval_path = args.eval_path
    if eval_path is None:
        if eval_type == "diffusion":
            eval_path = "models/co3d_diffusion"
        elif eval_type == "regression":
            eval_path = "models/co3d_regression"

    if eval_type == "diffusion":
        evaluate_ray_diffusion(eval_path)
        # process_predictions(eval_path, 7)
    elif eval_type == "regression":
        evaluate_ray_regression(eval_path)
        process_predictions(eval_path, 0)
    else:
        raise Exception(f"Unknown eval_type: {eval_type}")
