#!/bin/bash
visdom &
pip install -r requirements.txt | grep -v "already satisfied"
coverage run --source=abp,abp/examples -m unittest discover -v
echo
echo "Line Coverage:"
coverage report | tail -2
kill %1
