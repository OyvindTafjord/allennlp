#! /usr/bin/env python

"""
Basic script for generating multiple AllenNLP train commands for a given
config.jsonnet file and a jsonl file with parameter substitutions.
"""

import json
import re
import os
import subprocess
import sys
import argparse


def convert_config(old_content, out_file, new_params):
    content = old_content
    for key, value in new_params.items():
        if isinstance(value, str):
            value = f'"{value}"'
        content = re.sub(fr'(local {key}\s*=\s*).*?;', fr'\g<1>{value};', content)
    open(out_file, "w").write(content)
    return out_file


def main(config, replace, outdir, shfile, index, experiment) -> None:
    out_content = ["#!/bin/bash"]
    with open(replace, 'r') as file:
        replaces = [json.loads(line.strip()) for line in file]
    old_config = open(config, 'r').read()
    config_base = os.path.basename(config)
    config_split = os.path.splitext(config_base)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if experiment is not None:
        old_experiment = open(experiment, 'r').read()
        experiment_base = os.path.basename(experiment)
        experiment_split = os.path.splitext(experiment_base)
    for new_params in replaces:
        id = new_params.get('hsid', index)
        new_file = os.path.join(outdir, f'{config_split[0]}-multi{id}{config_split[1]}')
        new_dir = os.path.join(outdir, f'multi{index}')
        convert_config(old_config, new_file, new_params)
        if experiment is not None:
            config_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {new_file}', shell=True, universal_newlines=True).strip()
            new_experiment = re.sub(r'\$\$DS_CONFIG\$\$', config_dataset_id, old_experiment)
            new_experiment = re.sub(r'\$\$HSID\$\$', f'multi{id}', new_experiment)
            new_experiment_name = f'{experiment_split[0]}-multi{id}'
            new_experiment_file = os.path.join(outdir, new_experiment_name + ".yml")
            open(new_experiment_file, "w").write(new_experiment)
            out_content.append(f"beaker experiment create -n {new_experiment_name} -f {new_experiment_file}")
        else:
            out_content.append(f'allennlp/run.py train -s {new_dir} {new_file}')
        index += 1
    open(shfile, "w").write("\n".join(out_content))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate run-many-train.sh to sequentially run AllenNLP train.")
    parser.add_argument('--config', type=str, required=True, help='Original config file.')
    parser.add_argument('--replace', type=str, required=True, help='Replacement jsonl file.')
    parser.add_argument('--outdir', type=str, required=True, help='Output root directory')
    parser.add_argument('--experiment', type=str, default=None, required=False, help='Beaker experiment yaml file')
    parser.add_argument('--shfile', type=str, default="run-many-train.sh", help="Generated shell file.")
    parser.add_argument('--index', type=int, default=0, help="Start index for generated variations.")
    args = parser.parse_args()
    main(args.config, args.replace, args.outdir, args.shfile, args.index, args.experiment)
