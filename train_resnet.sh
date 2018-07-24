python ./tools/train_resnet.py --gpu 0\
    --solver models/ResNet/solver.prototxt\
    --weights models/ResNet/ResNet-101-model.caffemodel\
    --imdb voc_2007_trainval\
    --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml
