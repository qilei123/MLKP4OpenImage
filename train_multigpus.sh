python ./tools/train_net_multi_gpu.py --gpu 0,1\
        --solver models/VGG16/solver.prototxt\
        --weights data/ImageNet_models/VGG16_faster_rcnn_final.caffemodel\
        --imdb voc_2007_trainval\
        --cfg experiments/cfgs/faster_rcnn_end2end.yml
