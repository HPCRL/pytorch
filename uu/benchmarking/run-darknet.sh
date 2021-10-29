NTIME=1

echo "========= 512 "
python darknet19-ref-no-cp.py 512 $NTIME
python darknet19-ref-sq-cp.py 512 $NTIME
echo "==========ours "
python darknet19-our.py 512 1 $NTIME
python darknet19-our.py 512 2 $NTIME
python darknet19-our.py 512 4 $NTIME
python darknet19-our.py 512 8 $NTIME
python darknet19-our.py 512 16 $NTIME


echo "========= 1024 "
python darknet19-ref-no-cp.py 1024 $NTIME
python darknet19-ref-sq-cp.py 1024 $NTIME
echo "==========ours "
python darknet19-our.py 1024 1 $NTIME
python darknet19-our.py 1024 2 $NTIME
python darknet19-our.py 1024 4 $NTIME
python darknet19-our.py 1024 8 $NTIME
python darknet19-our.py 1024 16 $NTIME


echo "========= 2048 "
python darknet19-ref-no-cp.py 2048 $NTIME
python darknet19-ref-sq-cp.py 2048 $NTIME
echo "==========ours "
python darknet19-our.py 2048 1 $NTIME
python darknet19-our.py 2048 2 $NTIME
python darknet19-our.py 2048 4 $NTIME
python darknet19-our.py 2048 8 $NTIME
python darknet19-our.py 2048 16 $NTIME


echo "========= 3072 "
python darknet19-ref-no-cp.py 3072 $NTIME
python darknet19-ref-sq-cp.py 3072 $NTIME
echo "==========ours "
python darknet19-our.py 3072 1 $NTIME
python darknet19-our.py 3072 2 $NTIME
python darknet19-our.py 3072 4 $NTIME
python darknet19-our.py 3072 8 $NTIME
python darknet19-our.py 3072 16 $NTIME



echo "========= 4096 "
python darknet19-ref-no-cp.py 4096 $NTIME
python darknet19-ref-sq-cp.py 4096 $NTIME
echo "==========ours "
python darknet19-our.py 4096 1 $NTIME
python darknet19-our.py 4096 2 $NTIME
python darknet19-our.py 4096 4 $NTIME
python darknet19-our.py 4096 8 $NTIME
python darknet19-our.py 4096 16 $NTIME

echo "========= 5120 "
python darknet19-ref-no-cp.py 5120 $NTIME
python darknet19-ref-sq-cp.py 5120 $NTIME
echo "==========ours "
python darknet19-our.py 5120 1 $NTIME
python darknet19-our.py 5120 2 $NTIME
python darknet19-our.py 5120 4 $NTIME
python darknet19-our.py 5120 8 $NTIME
python darknet19-our.py 5120 16 $NTIME

######### 2080
echo "========= 6144 "
python darknet19-ref-no-cp.py 6144 $NTIME
python darknet19-ref-sq-cp.py 6144 $NTIME
echo "==========ours "
python darknet19-our.py 6144 1 $NTIME
python darknet19-our.py 6144 2 $NTIME
python darknet19-our.py 6144 4 $NTIME
python darknet19-our.py 6144 8 $NTIME
python darknet19-our.py 6144 16 $NTIME


echo "========= 7168 "
python darknet19-ref-no-cp.py 7168 $NTIME
python darknet19-ref-sq-cp.py 7168 $NTIME
echo "==========ours "
python darknet19-our.py 7168 1 $NTIME
python darknet19-our.py 7168 2 $NTIME
python darknet19-our.py 7168 4 $NTIME
python darknet19-our.py 7168 8 $NTIME
python darknet19-our.py 7168 16 $NTIME


echo "========= 8192 "
python darknet19-ref-no-cp.py 8192 $NTIME
python darknet19-ref-sq-cp.py 8192 $NTIME
echo "==========ours "
python darknet19-our.py 8192 1 $NTIME
python darknet19-our.py 8192 2 $NTIME
python darknet19-our.py 8192 4 $NTIME
python darknet19-our.py 8192 8 $NTIME
python darknet19-our.py 8192 16 $NTIME

