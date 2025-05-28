import argparse, os, zipfile, random, sys, cv2, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array
from PIL import UnidentifiedImageError


SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
VAL_SPLIT  = 0.2

# read picture
def safe_load(path, target=IMG_SIZE):
    try:
        img = load_img(path, target_size=target)
        return img_to_array(img)[None, ...]
    except (UnidentifiedImageError, OSError):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"cannot read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target)
        return img.astype("float32")[None, ...]

# build mode
def build_model(num_classes, binary=False):
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.Rescaling(1./255)(x)

    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.Dropout(0.3)(x)

    if binary:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# access data
ap = argparse.ArgumentParser()
ap.add_argument("--train",   action="store_true", help="train model")
ap.add_argument("--zip",     help="zip archive (ImageFolder)")
ap.add_argument("--data_dir",default="dataset",   help="unzipped folder")
ap.add_argument("--epochs",  type=int, default=30)
ap.add_argument("--binary",  action="store_true", help="red vs other")
ap.add_argument("--predict", help="image file to classify")
args = ap.parse_args()

# train
if args.train:
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(args.data_dir)

    
    raw_train = keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = raw_train.class_names
    if len(class_names) < 2 and not args.binary:
        sys.exit(" one type warning")
    np.save("class_names.npy", class_names)

    raw_val = keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # ② add repeat / prefetch
    train_ds = raw_train.repeat().prefetch(tf.data.AUTOTUNE)
    val_ds   = raw_val.repeat().prefetch(tf.data.AUTOTUNE)

    # build
    model = build_model(len(class_names), binary=args.binary)
    loss  = "binary_crossentropy" if args.binary else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    ckpt = keras.callbacks.ModelCheckpoint(
        "best_colour_model.keras", save_best_only=True, monitor="val_accuracy"
    )
    es = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True
    )

    # training
    steps = len(raw_train)   
    vsteps= len(raw_val)
    model.fit(train_ds,
            epochs=args.epochs,
            steps_per_epoch=steps,
            validation_data=val_ds,
            validation_steps=vsteps,
            callbacks=[ckpt, es],
            verbose=2)

# predict
if args.predict:
    if not os.path.exists("best_colour_model.keras"):
        sys.exit(" best_colour_model.keras not found. please train first！")

    model = keras.models.load_model("best_colour_model.keras")
    class_names = np.load("class_names.npy", allow_pickle=True)

    arr  = safe_load(args.predict)
    prob = model.predict(arr)[0]
    if model.output_shape[-1] == 1:              # binary
        label = "red" if prob[0] > 0.5 else "other"
        conf  = prob[0] if prob[0] > 0.5 else 1-prob[0]
    else:                                        # multi-class
        idx   = int(prob.argmax())
        label = class_names[idx]
        conf  = prob[idx]
    print(f"{os.path.basename(args.predict)} → {label} ({conf*100:.2f}%)")
