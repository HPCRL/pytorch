bash run-vgg.sh 2>&1 | tee a100-vgg-reg.txt
bash run-darknet.sh 2>&1 | tee a100-darknet-reg.txt

bash large.sh 2>&1 | tee a100-large.txt
