#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Number of passed arguments is not equal to 1 (received $#)" >&2
    exit 1
fi

message="$1"

cd ~/workspace2/team9_ws
source ./my_env/bin/activate

python3 ./src/send_script/send_script/CommentReader.py "$message"