implementation of 
http://proceedings.mlr.press/v80/xie18c/xie18c.pdf
using pytorch

python 3.6
pytorch
numpy 

TODO: connect a dataset
Try other networks...


download office-31 dataset:
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE" -O office31.tar.gz && rm -rf /tmp/cookies.txt
tar -xzf office31.tar.gz

run on mnist-svhn dataset:
python main.py --epoch 10 --batch_size 128 --set_device cuda

run on office-31 dataset:
!python main.py --epoch 10 --batch_size 128 --set_device cuda --dataset office_31 --n_class 31

