# image-recognition
simple color and ball recognition for robocon
python colour_classifier.py \
      --mode train \
      --data ./datasets/colours \
      --epochs 50 \
      --batch 16 \
      --out model_colour.h5
