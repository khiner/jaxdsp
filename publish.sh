#!/bin/bash
rm -rf build dist
pip install twine
python setup.py sdist bdist_wheel
python -m twine upload dist/*
