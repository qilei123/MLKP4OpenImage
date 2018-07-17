python ./tools/train_net_multi_gpu_resnet.py --gpu 0,1\
        --solver models/ResNet/solver.prototxt\
        --weights /home/qileimail123/openimagev4/MLKP4OpenImage/models/ResNet/ResNet-101-model.caffemodel\
        --imdb voc_2007_trainval\
        --cfg experiments/cfgs/rfcn_end2end_ohem.yml
