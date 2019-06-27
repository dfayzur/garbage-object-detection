# Garbage object detection (Alpha)

![Example 1](https://github.com/maartensukel/garbage-object-detection/blob/master/examples/prediction_example_1.gif)

![Example 2](https://github.com/maartensukel/garbage-object-detection/blob/master/examples/prediction_example_2.gif)

A repository with the source code to train an object detection model on a small dataset of annotated trash related images. The annotated objects in this dataset are garbage containers, garbage bags and cardboard.

The garbage detection model has been made for the City of Amsterdam, and is freely available for reuse for any purpose.

The code to train the model is based on the code from [this repository](https://github.com/bourdakos1/Custom-Object-Detection/)

## Data

![Example 1](https://github.com/maartensukel/garbage-object-detection/blob/master/examples/annotation_example1.png)![Example 2](https://github.com/maartensukel/garbage-object-detection/blob/master/examples/annotation_example2.png)


The dataset consist of 994 images and 994 annotations. A total of 4111 objects is annotated.

# RECOUNT FROM XMLS
* 1843 containers
* 1069 garbage bags
* 1199 cardboard

## 1) Download the dataset

The dataset with the annotations and images can be downloaded [here](https://drive.google.com/file/d/1aT0EsE_DopAaTemeMiGenjesr9tUr75i/view?usp=sharing)

## 2) Install Tensorflow

To use Tensorflow it must first be [installed](https://www.tensorflow.org/install).

## 2) Install Tensorflow research models
For using tensorflow the tensorflow research models have to be downloaded, installed and added to the python path, for a instruction on how to do this go [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

Don't forget to add the models to the research path.

### 3) Create TensorFlow records

```bash
python object_detection/create_tf_record.py
```
### 4) Download a Base Model
Training an object detector from scratch can take days, even when using multiple GPUs! In order to speed up training, we’ll take an object detector trained on a different dataset, and reuse some of it’s parameters to initialize our new model.

You can find models to download from this [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Each model varies in accuracy and speed. I used `faster_rcnn_resnet101_coco` for the demo.

Extract the files and move all the `model.ckpt` to our models directory.


### 5) Train the Model
Run the following script to train the model:

```bash
python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=faster_rcnn_resnet101.config
```

### 6) Export the Inference Graph
The training time is dependent on the amount of training data. 

You can find checkpoints for your model in `garbage-object-detection/train`.

Move the model.ckpt files with the highest number to the root of the repo:
- `model.ckpt-STEP_NUMBER.data-00000-of-00001`
- `model.ckpt-STEP_NUMBER.index`
- `model.ckpt-STEP_NUMBER.meta`

In order to use the model, you first need to convert the checkpoint files (`model.ckpt-STEP_NUMBER.*`) into a frozen inference graph by running this command:

```bash
python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101.config \
        --trained_checkpoint_prefix train/model.ckpt-STEP_NUMBER\
        --output_directory output_inference_graph
```

You should see a new `output_inference_graph` directory with a `frozen_inference_graph.pb` file.

### 7) Test the Model
Just run the following command:

```bash
python object_detection/object_detection_runner.py
```

The files with the predictions will be saved in the 'output' folder. The images that are annotated can be found in the 'test_images' folder.