# NTIME=1
# echo "========= vgg large "

# echo "========= vgg 10k "
# python vgg16-our.py 10240 8 $NTIME
# python vgg16-our.py 10240 16 $NTIME
# echo "========= vgg 20k "
# python vgg16-our.py 20480 16 $NTIME
# python vgg16-our.py 20480 32 $NTIME



# echo "========= dark large "

# echo "========= d 10k "
# python darknet19-our.py 10240 8 $NTIME
# python darknet19-our.py 10240 16 $NTIME

# echo "========= d 20k "
# python darknet19-our.py 20480 16 $NTIME
# python darknet19-our.py 20480 32 $NTIME




echo "========= vgg 40k "
# python vgg16-our.py 40960 16 $NTIME
# python vgg16-our.py 40960 32 $NTIME
# python vgg16-our.py 40960 64 $NTIME
python vgg16-our.py 40960 128 $NTIME
python vgg16-our.py 40960 256 $NTIME
python vgg16-our.py 40960 512 $NTIME
echo "========= d 40k "
# python darknet19-our.py 40960 16 $NTIME
# python darknet19-our.py 40960 32 $NTIME
# python darknet19-our.py 40960 64 $NTIME
python darknet19-our.py 40960 128 $NTIME
python darknet19-our.py 40960 256 $NTIME
python darknet19-our.py 40960 512 $NTIME

# echo "========= vgg 80k "
# python vgg16-our.py 81920 32 $NTIME
# python vgg16-our.py 81920 64 $NTIME
# python vgg16-our.py 81920 128 $NTIME


# echo "========= d 80k "
# python darknet19-our.py 81920 32 $NTIME
# python darknet19-our.py 81920 64 $NTIME
# python darknet19-our.py 81920 128 $NTIME

# output of convchain too large 80x80x1kx4xC(512,1024)  12GiB and 24GiB 


# echo "========= 80k 256"
# python vgg16-our.py 81920 256 $NTIME
# python darknet19-our.py 81920 256 $NTIME

# echo "========= 80k 512"
# python vgg16-our.py 81920 512 $NTIME
# python darknet19-our.py 81920 512 $NTIME

# echo "========= vgg 40k "
# python vgg16-our.py 40960 16 $NTIME
# python vgg16-our.py 40960 32 $NTIME
# python vgg16-our.py 40960 64 $NTIME
# echo "========= d 40k "
# python darknet19-our.py 40960 16 $NTIME
# python darknet19-our.py 40960 32 $NTIME
# python darknet19-our.py 40960 64 $NTIME



# echo "========= vgg 80k "
# python vgg16-our.py 81920 32 $NTIME
# python vgg16-our.py 81920 64 $NTIME
# python vgg16-our.py 81920 128 $NTIME

# echo "========= d 80k "
# python darknet19-our.py 81920 32 $NTIME
# python darknet19-our.py 81920 64 $NTIME
# python darknet19-our.py 81920 128 $NTIME

