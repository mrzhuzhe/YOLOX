# https://www.kaggle.com/drzhuzhe/carbon-fiber-defect-detect-demo

# https://www.kaggle.com/drzhuzhe/fiber-defect-detect

# python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 32 --fp16 -o -c ../reef-data/out/weights/yolox/yolox_s.pth
# python tools/train.py -f exps/example/custom/yolox_x.py -d 1 -b 32 --fp16 -o -c ../reef-data/out/weights/yolox/yolox_x.pth


# python tools/eval.py -f  exps/example/custom/yolox_s.py -c ./YOLOX_outputs/yolox_s/latest_ckpt.pth  -b 32 -d 1 --conf 0.1 --fp16 --fuse  
