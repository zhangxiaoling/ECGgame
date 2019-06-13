#!/bin/bash
echo "downloading model,please wait"
git clone https://github.com/zhangxiaoling/Zhang-s-Sisters

echo "==== running entry script on the test set ===="
# Clear previous answers.txt
rm -f answers.txt
# Generate new answers.txt
python result.py
echo "=================== Done! ===================="
