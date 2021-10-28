echo "========= 512 "
python vgg16-ref-no-cp.py 512
python vgg16-ref-sq-cp.py 512
echo "==========ours "
python vgg16-our.py 512 1
python vgg16-our.py 512 2
python vgg16-our.py 512 4
python vgg16-our.py 512 8
python vgg16-our.py 512 16


echo "========= 1024 "
python vgg16-ref-no-cp.py 1024
python vgg16-ref-sq-cp.py 1024
echo "==========ours "
python vgg16-our.py 1024 1
python vgg16-our.py 1024 2
python vgg16-our.py 1024 4
python vgg16-our.py 1024 8
python vgg16-our.py 1024 16


echo "========= 2048 "
python vgg16-ref-no-cp.py 2048
python vgg16-ref-sq-cp.py 2048
echo "==========ours "
python vgg16-our.py 2048 1
python vgg16-our.py 2048 2
python vgg16-our.py 2048 4
python vgg16-our.py 2048 8
python vgg16-our.py 2048 16


echo "========= 3072 "
python vgg16-ref-no-cp.py 3072
python vgg16-ref-sq-cp.py 3072
echo "==========ours "
python vgg16-our.py 3072 1
python vgg16-our.py 3072 2
python vgg16-our.py 3072 4
python vgg16-our.py 3072 8
python vgg16-our.py 3072 16






