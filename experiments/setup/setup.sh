#!/bin/bash

current_dir=${PWD##*/}

if [[ $current_dir != "rffnet" ]]; then
    echo "Scripts must be run from the directory 'rffnet'"
    exit
fi

declare -a subdirs=("eval/simulated_benchmarks" "eval/real_world_benchmarks" "figures" "tables")
for subdir in "${subdirs[@]}"
do 
    cmd="mkdir -p $subdir"
    echo "$cmd"; $cmd
done