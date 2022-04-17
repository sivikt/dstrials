"""
Just runs TF object detector model over a set of images and benchmark GPU performance.
"""

import cv2
import time
import math
from pathlib import Path


ITER_NUM=3
IMAGE_WIDTH=500
IMAGE_HEIGHT=500
BATCH_SIZE=1


INPUT_DIR = Path().cwd() / 'data' / 'image'


images_paths = list(INPUT_DIR.glob('**/*.jpg'))
print('Total %d images' % len(images_paths))

MAX_IMAGES = 2**int(math.log(len(images_paths),2)-3)
print('Try to init %d images' % MAX_IMAGES)
#MAX_IMAGES = 16

IMAGES = [cv2.imread(str(img_path)) for img_path in images_paths[:MAX_IMAGES]]

print('%d images are loaded into memory' % len(IMAGES))



import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


MODEL_NAME = 'object_detector_16-classes'
PATH_TO_FROZEN_GRAPH = Path().cwd() / (MODEL_NAME + '_frozen_graph.pb')


def test_tf(images, max_images=1, iter_num=1, out_f_path=None):
    print("LOADING PERSISTED MODELS FROM ", str(PATH_TO_FROZEN_GRAPH))

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(PATH_TO_FROZEN_GRAPH), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    print("LOADED", str(PATH_TO_FROZEN_GRAPH))

    
    results = None
    
    out_f=None
    if out_f_path is not None:
        out_f = open(out_f_path, 'w')
    
    for n in [1 << i for i in range(int(math.log(max_images,2))+1)]:
        images_to_test = images[:n]
        sum_time = 0
        for i in range(iter_num):
            all_inf_start = time.time()
            results = test_tf_inference(images_to_test, detection_graph)
            all_inf_end = time.time()
            test_time = all_inf_end - all_inf_start
            sum_time += test_time

        msg = 'test_tf: %d images: %.3f ms' % (n, (sum_time/iter_num)*1000)
        print(msg)
        if out_f is not None:
            out_f.write(msg)
            out_f.write('\n')
            out_f.flush()
    
    results_out = []
    for res in results:
        img_objects = []
        
        for i in range(res['num_detections']):
            img_objects.append({
                'score': round(res['detection_scores'][i],2),
                'cls': res['detection_classes'][i],
                'xmin': round(res['detection_boxes'][i][1],2),
                'ymin': round(res['detection_boxes'][i][0],2),
                'xmax': round(res['detection_boxes'][i][3],2),
                'ymax': round(res['detection_boxes'][i][2],2),
            })
        
        img_objects.sort(key=lambda v: v['cls'])
        results_out.append(img_objects)    
    
    assert len(results_out)==n, '%d != %d' % (len(results_out), n)
    
    return results_out


def test_tf_inference(images, detection_graph):
    results = []
    
    sz = len(images)
    batches = [images[lo:min(lo+BATCH_SIZE, sz)] for lo in range(0, sz, BATCH_SIZE)]

    
    ops = detection_graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}

    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    
    with detection_graph.as_default():
        with tf.Session() as sess:
            for batch in batches:
                batch = np.array([cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)) for img in batch])

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: batch})

                for i in range(len(output_dict['num_detections'])):
                    img_res = {}
                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    img_res['num_detections'] = int(output_dict['num_detections'][i])
                    img_res['detection_classes'] = output_dict['detection_classes'][i].astype(np.uint8)
                    img_res['detection_boxes'] = output_dict['detection_boxes'][i]
                    img_res['detection_scores'] = output_dict['detection_scores'][i]

                    results.append(img_res)
        
    return results


tf_results = test_tf(IMAGES, max_images=MAX_IMAGES, iter_num=ITER_NUM, out_f_path='test-results2.log')
