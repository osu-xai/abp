#!/bin/bash
pip install -r requirements.txt | grep -v "already satisfied"
coverage run --source=abp,abp/examples -m unittest discover -v
coverage report | (head -1 && tail -1)
