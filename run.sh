#! /bin/bash
echo 'downloading model,please wait'
git clone https://github.com/zhangxiaoling/Zhang-s-Sisters_GPU

echo "==== running entry script on the test set ===="
rm -f answers.txt
python result_gpu.py
echo "=================== Done! ===================="
