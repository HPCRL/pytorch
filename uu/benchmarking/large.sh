NTIME=1
echo "========= vgg large "
python vgg16-our.py 10240 16 $NTIME
python vgg16-our.py 20480 32 $NTIME




echo "========= dark large "
python darknet19-our.py 10240 16 $NTIME
python darknet19-our.py 20480 32 $NTIME
