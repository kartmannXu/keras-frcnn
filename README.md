# keras-frcnn
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.
cloned from https://github.com/yhenon/keras-frcnn/


USAGE:
- Both theano and tensorflow backends are supported. However compile times are very high in theano, and tensorflow is highly recommended.
- `main.py` can be used to train and test a model. To train on Pascal VOC data, simply do:
`python main.py --mode train -p /path/to/pascalvoc/`. 
- the Pascal VOC data set (images and annotations for bounding boxes around the classified objects) can be obtained from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- simple_parser provides an alternative way to input data, using a text file. Simply provide a text file, with each
line containing:

    `filepath,x1,y1,x2,y2,class_name`

    For example:

    /data/imgs/img_001.jpg,837,346,981,456,cow
    
    /data/imgs/img_002.jpg,215,312,279,391,cat

    The classes will be inferred from the file. To use the simple parser instead of the default pascal voc style parser,
    use the command line option `-o simple`. For example `python main.py --mode train -o simple -p annotation.txt`.

- Training will write weights to disk to an hdf5 file, as well as all the setting of the training run to a `pickle` file. These
settings can then be loaded in testing part. Thus in testing, only these 4 arguments are required: 

    - --mode, --config_filename, --path, --num_rois

- Test mode in `main.py` can be used to perform inference, given pretrained weights and a config file. Specify a path to the folder containing
images:
    `python main.py --mode test -p /path/to/test_data/ -c /path/to/config_file.pickle`
- Data augmentation can be applied by specifying `--hf` for horizontal flips, `--vf` for vertical flips and `--rot` for 90 degree rotations
- In training, the value of `output_weight_path` must contains a pair of empty brace, you could refer to the default value: `weights\\model_frcnn_{}`.
- In testing, the value of `input_weight_path` is __NOT__ the file name, it should be the prefix of the weights files output by training process, 
    without classifier/rpn.h5 suffix. for example, if the weight files are `weights\model_frcnn_resnet50classifier.h5` and
    `weights\model_frcnn_resnet50rpn.h5`, then the value of `input_weight_path` should be `weights\model_frcnn_resnet50`.
- The value of `--path` in training should be the path to annotation file, e.g. `annotation.txt`, while in testing the value 
    should be a directory that includes the images to be tested. 

SIMPLE EXAMPLE:
- Train: `python main.py --mode train -o simple -p annotation.txt -hf -rot -c config_own.pickle`
- Testing: `python main.py --mode train -o simple -p Image\\001`

NOTES:
- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].
- The theano backend by default uses a 7x7 pooling region, instead of 14x14 as in the frcnn paper. This cuts down compiling time slightly.
- The tensorflow backend performs a resize on the pooling region, instead of max pooling. This is much more efficient and has little impact on results.
