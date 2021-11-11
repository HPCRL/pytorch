bash run-vgg.sh 2>&1 | tee a100-vgg-reg-our.txt
bash run-darknet.sh 2>&1 | tee a100-darknet-reg-our.txt

bash run-vgg.sh 2>&1 | tee a100-vgg-reg1-our.txt
bash run-darknet.sh 2>&1 | tee a100-darknet-reg1-our.txt

bash run-vgg.sh 2>&1 | tee a100-vgg-reg2-our.txt
bash run-darknet.sh 2>&1 | tee a100-darknet-reg2-our.txt


#bash large.sh 2>&1 | tee a100-large.txt
