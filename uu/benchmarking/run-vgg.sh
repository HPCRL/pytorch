NTIME=1

echo "========= 512 "
python vgg16-ref-no-cp.py 512 $NTIME
python vgg16-ref-sq-cp.py 512 $NTIME
echo "==========ours "
python vgg16-our.py 512 1 $NTIME
python vgg16-our.py 512 2 $NTIME
python vgg16-our.py 512 4 $NTIME
python vgg16-our.py 512 8 $NTIME
python vgg16-our.py 512 16 $NTIME


echo "========= 1024 "
python vgg16-ref-no-cp.py 1024 $NTIME
python vgg16-ref-sq-cp.py 1024 $NTIME
echo "==========ours "
python vgg16-our.py 1024 1 $NTIME
python vgg16-our.py 1024 2 $NTIME
python vgg16-our.py 1024 4 $NTIME
python vgg16-our.py 1024 8 $NTIME
python vgg16-our.py 1024 16 $NTIME


echo "========= 2048 "
python vgg16-ref-no-cp.py 2048 $NTIME
python vgg16-ref-sq-cp.py 2048 $NTIME
echo "==========ours "
python vgg16-our.py 2048 1 $NTIME
python vgg16-our.py 2048 2 $NTIME
python vgg16-our.py 2048 4 $NTIME
python vgg16-our.py 2048 8 $NTIME
python vgg16-our.py 2048 16 $NTIME


echo "========= 3072 "
python vgg16-ref-no-cp.py 3072 $NTIME
python vgg16-ref-sq-cp.py 3072 $NTIME
echo "==========ours "
python vgg16-our.py 3072 1 $NTIME
python vgg16-our.py 3072 2 $NTIME
python vgg16-our.py 3072 4 $NTIME
python vgg16-our.py 3072 8 $NTIME
python vgg16-our.py 3072 16 $NTIME

###2080


echo "========= 4096 "
python vgg16-ref-no-cp.py 4096 $NTIME
python vgg16-ref-sq-cp.py 4096 $NTIME
echo "==========ours "
python vgg16-our.py 4096 1 $NTIME
python vgg16-our.py 4096 2 $NTIME
python vgg16-our.py 4096 4 $NTIME
python vgg16-our.py 4096 8 $NTIME
python vgg16-our.py 4096 16 $NTIME






