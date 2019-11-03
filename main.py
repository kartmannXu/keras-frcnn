from __future__ import division
import pprint
import sys
import time
from optparse import OptionParser
import pickle

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import generic_utils
from utils import *

sys.setrecursionlimit(40000)


def parse_args():
    parser = OptionParser()

    parser.add_option("--mode", type=str, dest="mode", help="Code Running mode, must be one of `train` or `test`.")
    parser.add_option("-p", "--path", dest="path", help="Path to training data.")
    parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                      default="pascal_voc")
    parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.",
                      default=32)
    parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                      default='resnet50')
    parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false)",
                      action="store_true", default=False)
    parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                      action="store_true", default=False)
    parser.add_option("--rot", "--rot_90", dest="rot_90",
                      help="Augment with 90 degree rotations in training. (Default=false).",
                      action="store_true", default=False)
    parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
    parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to"
                                                                        " the training (to be used when testing).",
                      default="config.pickle")
    parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
                      default='weights\\model_frcnn_{}')
    parser.add_option("--input_weight_path", dest="input_weight_path",
                      help="Input path for weights. If not specified, will try to load default weights provided by keras.")
    parser.add_option("--epoch_length", dest="epoch_length", type=int, default=1000, help="Iterations per Epoch.")
    options, args = parser.parse_args()

    if not options.path:  # if filename is not given
        parser.error('Error: path to training data must be specified. Pass --path to command line')
    if options.mode not in ["train", "test"]:
        parser.error('Error: mode must be one of `train` or `test`.')
    return options


def train_frcnn(options):
    if options.parser == 'pascal_voc':
        from utils import voc_parser as get_data
    elif options.parser == 'simple':
        from utils import simple_parser as get_data
    else:
        raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

    # pass the settings from the command line, and persist them in the config object
    C = Config()

    C.use_horizontal_flips = bool(options.horizontal_flips)
    C.use_vertical_flips = bool(options.vertical_flips)
    C.rot_90 = bool(options.rot_90)

    C.model_path = options.output_weight_path.format(options.network)
    C.num_rois = int(options.num_rois)

    if options.network == 'resnet50':
        C.network = 'resnet50'
        from utils import rpn_res as rpn
        from utils import classifier_res as classifier_func
        from utils import get_img_output_length_res as get_img_output_length
        from utils import nn_base_res as nn_base
    elif options.network == 'vgg':
        C.network = 'vgg'
        from utils import rpn_vgg as rpn
        from utils import classifier_vgg as classifier_func
        from utils import get_img_output_length_vgg as get_img_output_length
        from utils import nn_base_vgg as nn_base
    else:
        print('Not a valid model')
        raise ValueError

    # check if weight path was passed via command line
    if options.input_weight_path:
        C.base_net_weights = options.input_weight_path
    else:
        # set the path to weights based on backend and model
        C.base_net_weights = get_weight_path(options.network)

    all_imgs, classes_count, class_mapping = get_data(options.path)

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping

    inv_map = {v: k for k, v in class_mapping.items()}

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))

    config_output_filename = options.config_filename

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            config_output_filename))
    #
    random.shuffle(all_imgs)

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    data_gen_train = get_anchor_gt(train_imgs, classes_count, C, get_img_output_length, K.backend(), mode='train')
    data_gen_val = get_anchor_gt(val_imgs, classes_count, C, get_img_output_length, K.backend(), mode='val')

    if K.backend() == "theano":
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = rpn(shared_layers, num_anchors)

    classifier = classifier_func(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    try:
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights + "rpn.h5", by_name=True)
        model_classifier.load_weights(C.base_net_weights + "classifier.h5", by_name=True)
    except Exception as e:
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
        print('Exception: {}'.format(e))

    optimizer = Adam(lr=1e-5, decay=2e-7)
    optimizer_classifier = Adam(lr=1e-5, decay=2e-7)

    model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[class_loss_cls, class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = options.epoch_length
    num_epochs = int(options.num_epochs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    print('Starting training')

    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                        mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. '
                              'Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)
                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.backend(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []
                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(selected_neg_samples)
                    else:
                        sel_samples = random.choice(selected_pos_samples)

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                              ('detector_cls', losses[iter_num, 2]),
                                              ('detector_regr', losses[iter_num, 3])])

                iter_num += 1

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if C.verbose:
                            print(f'Total loss decreased from {best_loss:.3f} to {curr_loss:.3f}, saving weights to '
                                  f'{C.model_path}')
                        best_loss = curr_loss
                        model_classifier.save_weights(C.model_path + "classifier.h5")
                        model_rpn.save_weights(C.model_path + "rpn.h5")
                    break
            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    print('Training complete, exiting.')


def test_frcnn(options):
    config_output_filename = options.config_filename

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    if C.network == 'resnet50':
        from utils import rpn_res as rpn
        from utils import classifier_res as classifier_func
        from utils import nn_base_res as nn_base
    elif C.network == 'vgg':
        from utils import rpn_vgg as rpn
        from utils import classifier_vgg as classifier_func
        from utils import nn_base_vgg as nn_base

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    img_path = options.path

    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = int(options.num_rois)

    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'vgg':
        num_features = 512

    if K.backend() == 'theano':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn(shared_layers, num_anchors)

    classifier = classifier_func(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn, model_classifier = load_weights_frcnn(model_rpn, model_classifier, C)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    bbox_threshold = 0.8

    if not os.path.exists("results_imgs"):
        os.mkdir("results_imgs")

    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        st = time.time()
        filepath = os.path.join(img_path, img_name)

        img = cv2.imread(filepath)

        X, ratio = format_img(img, C)

        if K.backend() == 'tensorflow':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, C, K.backend(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              2)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk], [x1, y1, x2, y2]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        cv2.imwrite('.\\results_imgs\\{}.png'.format(idx), img)


if __name__ == '__main__':
    options = parse_args()
    if options.mode == "train":
        train_frcnn(options)
    else:
        test_frcnn(options)
