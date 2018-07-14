python ./tools/train_net_multi_gpu.py --gpu 0,1\
        --solver models/ResNet/solvecd r.prototxt\
        --weights /home/qileimail123/Development/openimagev4/CVDF/resnet101_faster_rcnn_bn_scale_merged_end2end_ohem_iter_70000.caffemodel\
        --imdb voc_2007_trainval\
        --cfg experiments/cfgs/rfcn_end2end_ohem.yml
