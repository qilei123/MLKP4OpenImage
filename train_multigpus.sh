python ./tools/train_net_multi_gpu.py --gpu 2,3\
        --solver models/VGG16/solver.prototxt\
        --weights output/faster_rcnn_end2end/voc_2007_trainvalls/vgg16_mlkp_iter_40000.caffemodel\
        --imdb voc_2007_trainval\
        --cfg experiments/cfgs/faster_rcnn_end2end.yml
