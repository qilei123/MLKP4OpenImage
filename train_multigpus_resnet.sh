python ./tools/train_net_multi_gpu_resnet.py --gpu 0,1\
        --solver models/ResNet/solver.prototxt\
        --weights /home/qileimail123/Development/openimagev4/CVDF/ResNet-101-model.caffemodel\
        --imdb openimagev4\
        --cfg experiments/cfgs/rfcn_end2end_ohem.yml
