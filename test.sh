#!/bin/bash
pip install -r requirements.txt
coverage run --source=abp,abp/examples -m unittest discover
coverage report
