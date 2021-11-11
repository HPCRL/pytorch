NTIME=1

echo "========= 512 "

echo "==========ours "
python vgg16-our.py 512 1 $NTIME
python vgg16-our.py 512 2 $NTIME
python vgg16-our.py 512 4 $NTIME
python vgg16-our.py 512 8 $NTIME
python vgg16-our.py 512 16 $NTIME


echo "========= 1024 "

echo "==========ours "
python vgg16-our.py 1024 1 $NTIME
python vgg16-our.py 1024 2 $NTIME
python vgg16-our.py 1024 4 $NTIME
python vgg16-our.py 1024 8 $NTIME
python vgg16-our.py 1024 16 $NTIME


echo "========= 2048 "

echo "==========ours "
python vgg16-our.py 2048 1 $NTIME
python vgg16-our.py 2048 2 $NTIME
python vgg16-our.py 2048 4 $NTIME
python vgg16-our.py 2048 8 $NTIME
python vgg16-our.py 2048 16 $NTIME


echo "========= 3072 "

echo "==========ours "
python vgg16-our.py 3072 1 $NTIME
python vgg16-our.py 3072 2 $NTIME
python vgg16-our.py 3072 4 $NTIME
python vgg16-our.py 3072 8 $NTIME
python vgg16-our.py 3072 16 $NTIME

###2080


echo "========= 4096 "

echo "==========ours "
python vgg16-our.py 4096 1 $NTIME
python vgg16-our.py 4096 2 $NTIME
python vgg16-our.py 4096 4 $NTIME
python vgg16-our.py 4096 8 $NTIME
python vgg16-our.py 4096 16 $NTIME



echo "========= 5120 "

echo "==========ours "
python vgg16-our.py 5120 1 $NTIME
python vgg16-our.py 5120 2 $NTIME
python vgg16-our.py 5120 4 $NTIME
python vgg16-our.py 5120 8 $NTIME
python vgg16-our.py 5120 16 $NTIME


echo "========= 6144 "

echo "==========ours "
python vgg16-our.py 6144 1 $NTIME
python vgg16-our.py 6144 2 $NTIME
python vgg16-our.py 6144 4 $NTIME
python vgg16-our.py 6144 8 $NTIME
python vgg16-our.py 6144 16 $NTIME


echo "========= 7168 "

echo "==========ours "
python vgg16-our.py 7168 1 $NTIME
python vgg16-our.py 7168 2 $NTIME
python vgg16-our.py 7168 4 $NTIME
python vgg16-our.py 7168 8 $NTIME
python vgg16-our.py 7168 16 $NTIME


