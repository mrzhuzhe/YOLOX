# Log

https://www.kaggle.com/drzhuzhe/carbon-fiber-defect-detect-demo
https://www.kaggle.com/drzhuzhe/fiber-defect-detect

# train
1. python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 32 --fp16 -o -c ../reef-data/out/weights/yolox/yolox_s.pth
2. python tools/train.py -f exps/example/custom/yolox_m.py -d 1 -b 32 --fp16 -o -c ../reef-data/out/weights/yolox/yolox_m.pth
3. python tools/train.py -f exps/example/custom/yolox_l.py -d 1 -b 32 --fp16 -o -c ../reef-data/out/weights/yolox/yolox_l.pth
4. python tools/train.py -f exps/example/custom/yolox_x.py -d 1 -b 32 --fp16 -o -c ../reef-data/out/weights/yolox/yolox_x.pth
## new
python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 32 --fp16 -o -c ../weights/yolox/yolox_s.pth

# eval
> python tools/eval.py -f  exps/example/custom/yolox_l.py -c ./YOLOX_outputs/yolox_l/yoloxl-best-640_ckpt.pth  -b 32 -d 1 --conf 0.01 --fp16 --fuse

# vis
> python tools/demo.py image  -f  exps/example/custom/yolox_s.py -c ./YOLOX_outputs/yolox_s/best_ckpt.pth --path ../reef-data/out/fold2/valid/images --nms 0.45 --tsize 1280 --conf 0.1 --fp16 --save_result --device gpu       
> python tools/demo.py image  -f  exps/example/custom/yolox_l.py -c ./YOLOX_outputs/yolox_l/yoloxl-best-640_ckpt.pth --path ../reef-data/out/fold2/valid/images --nms 0.45 --tsize 736 --conf 0.1 --fp16 --save_result --device gpu

# trick
1. tracking https://www.kaggle.com/parapapapam/yolox-inference-tracking-on-cots-lb-0-539