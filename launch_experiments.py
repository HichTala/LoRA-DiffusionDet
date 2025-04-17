import argparse
import json
import os.path
import subprocess

slurm_script = r"""#!/bin/bash
#SBATCH --job-name={job_name}        # name of job
#SBATCH -A mvq@v100
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
## SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
#SBATCH --partition=gpu_p2           # uncomment for gpu_p2 partition (32GB V100 GPU)
##SBATCH --partition=gpu_p4          # uncomment for gpu_p4 partition (40GB A100 GPU)
##SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)
# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
##SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --cpus-per-task=3           # number of cores per task for gpu_p2 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=6           # number of cores per task for gpu_p4 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=8           # number of cores per task for gpu_p5 (1/8 of 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logs/gpu_single%j.out    # name of output file
#SBATCH --error=logs/gpu_single%j.out     # name of error file (here, in common with the output file)
#SBATCH --qos=qos_gpu-t3

module purge # purge modules inherited by default
module load pytorch-gpu/py3/2.5.0 # load modules

export PYTHONUSERBASE=$WORK/.local
export WANDB_MODE=offline
export NO_ALBUMENTATIONS_UPDATE=1
set -x # activate echo of launched commands

srun python -u run_object_detection.py{command}"""


def get_args_parser():
    parser = argparse.ArgumentParser('Launch experients')

    parser.add_argument('--config', type=str, default="configs/models/diffusiondet_dota.json")
    parser.add_argument('--dataset_names', nargs='+', default=["detection-datasets/coco"])
    parser.add_argument('--seed', nargs='+', default=["1338"])
    parser.add_argument('--shots', nargs='+', default=["10"])
    parser.add_argument('--output_dir', type=str, default="diffusiondet")

    parser.add_argument('--freeze_mode', type=bool, default=False)
    parser.add_argument('--freeze_modules', type=str)
    parser.add_argument('--freeze_at', type=str)

    parser.add_argument('--use_lora', type=bool, default=False)
    parser.add_argument('--over_lora', type=bool, default=False)
    parser.add_argument('--lora_ranks', nargs='+', default=["8"])

    parser.add_argument('--exec_type', type=str, default="slurm")

    return parser


def build_cmd(config):
    cmd = ""
    for key, value in config.items():
        if key not in ['freeze_modules', 'freeze_at'] or value != '':
            cmd += f" --{key} "
            cmd += str(value)
    return cmd


def submit_job(cmd, exec_type, **kwargs):
    if exec_type == "slurm":
        formated_script = slurm_script.format(job_name=kwargs['shot'] + 'nl' + kwargs['seed'], command=cmd)
        with open('launchers/automatic_launcher.slurm', 'w') as f:
            f.write(formated_script)
        print('job_name', kwargs['shot'] + 'nl' + kwargs['seed'])
        return subprocess.call(['sbatch', 'launchers/automatic_launcher.slurm'])
    elif exec_type == "python":
        cmd = f"python run_object_detection.py{cmd}"
        return subprocess.run(cmd, shell=True)
    return None


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    if not args.freeze_modules:
        args.freeze_modules = ['', 'backbone', 'backbone', 'bias', 'norm']
    if not args.freeze_at:
        args.freeze_at = ['', '0', 'half', '', '']

    for dataset_name in args.dataset_names:
        for shot in args.shots:
            for seed in args.seed:
                if args.freeze_mode:
                    for freeze_modules, freeze_at in zip(args.freeze_modules, args.freeze_at):
                        output_dir = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/seed_{seed}/"

                        if len(freeze_modules) == 0:
                            output_dir += "full_finetuning"
                        output_dir += f"{freeze_modules}-{freeze_at}" if freeze_at != '0' else f"{freeze_modules}-full"
                        if output_dir[-1] == '-':
                            output_dir = output_dir[:-1]

                        config['freeze_modules'] = freeze_modules
                        config['freeze_at'] = freeze_at

                        config["dataset_name"] = dataset_name
                        config["seed"] = seed
                        config["shots"] = shot
                        config["output_dir"] = output_dir

                        cmd = build_cmd(config)
                        result = submit_job(cmd, exec_type=args.exec_type)
                        # if result.returncode != 0:
                        #     print(f"Error running command: python run_object_detection.py{cmd}")
                        #     return
                else:
                    output_dir = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/nolora/{seed}"

                    logging_steps = {'50': 970, '10':100, '5': 50, '1':10}

                    config["dataset_name"] = dataset_name
                    config["seed"] = seed
                    config["shots"] = shot
                    config["output_dir"] = output_dir
                    config["eval_steps"] = logging_steps[shot]
                    config["save_steps"] = logging_steps[shot]

                    if args.use_lora:
                        config["use_lora"] = True
                        for rank in args.lora_ranks:
                            config["lora_rank"] = rank
                            config["output_dir"] = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/lora/{seed}/{rank}"

                            if args.over_lora:
                                if os.path.isfile(f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/nolora/{seed}/trainer_state.json"):
                                    with open(f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/nolora/{seed}/trainer_state.json") as f:
                                        trainer_state = json.load(f)
                                    config['model_name_or_path'] = trainer_state['best_model_checkpoint']
                                    config["output_dir"] = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/overlora/{seed}/{rank}"

                            cmd = build_cmd(config)
                            result = submit_job(cmd, exec_type=args.exec_type, seed=seed, shot=shot)
                            if result != 0:
                                print(f"Error running command: python run_object_detection.py{cmd}")
                                return
                    else:
                        cmd = build_cmd(config)
                        result = submit_job(cmd, exec_type=args.exec_type, seed=seed, shot=shot)
                        if result != 0:
                            print(f"Error running command: python run_object_detection.py{cmd}")
                            return


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
