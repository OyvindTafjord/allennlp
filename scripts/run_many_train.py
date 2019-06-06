#! /usr/bin/env python

"""
Basic script for generating multiple AllenNLP train commands for a given
config.jsonnet file and a jsonl file with parameter substitutions.
"""

import json
import re
import os
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


def main(config, replace, outdir, shfile, index) -> None:
    out_content = ["#!/bin/bash"]
    with open(replace, 'r') as file:
        replaces = [json.loads(line.strip()) for line in file]
    old_config = open(config, 'r').read()
    config_base = os.path.basename(config)
    config_split = os.path.splitext(config_base)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for new_params in replaces:
        new_file = os.path.join(outdir, f'{config_split[0]}-multi{index}{config_split[1]}')
        new_dir = os.path.join(outdir, f'multi{index}')
        convert_config(old_config, new_file, new_params)
        out_content.append(f'allennlp/run.py train -s {new_dir} {new_file}')
        index += 1
    open(shfile, "w").write("\n".join(out_content))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate run-many-train.sh to sequentially run AllenNLP train.")
    parser.add_argument('--config', type=str, required=True, help='Original config file.')
    parser.add_argument('--replace', type=str, required=True, help='Replacement jsonl file.')
    parser.add_argument('--outdir', type=str, required=True, help='Output root directory')
    parser.add_argument('--shfile', type=str, default="run-many-train.sh", help="Generated shell file.")
    parser.add_argument('--index', type=int, default=0, help="Start index for generated variations.")
    args = parser.parse_args()
    main(args.config, args.replace, args.outdir, args.shfile, args.index)
