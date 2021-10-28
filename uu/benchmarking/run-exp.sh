bash run-vgg.sh 2>&1 | tee vgg-reg.txt
bash run-darknet.sh 2>&1 | tee darknet-reg.txt

bash large.sh 2>&1 | tee large.txt