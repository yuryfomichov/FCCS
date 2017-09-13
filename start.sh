#!/usr/bin/env bash

cd FCCS
source .env/bin/activate
git pull origin master
cd src
python3 start.py

echo "Started."

