python3 train.py --weights weights/yolov7-tiny.pt \
    		 --cfg cfg/training/yolov7-tiny.yaml \
    		 --data data/coco.yaml \
    		 --hyp data/hyp.scratch.tiny.yaml \
    		 --epochs 300 \
    		 --batch-size 8 \
    		 --img-size 640 \
    		 --device cpu \
    		 --project output \
    		 --name yolov7-tiny-coco-train
