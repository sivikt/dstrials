"""
Just runs TF object detector converted to OpenVINO format over a set of images and benchmark CPU performance.
"""

import cv2
import time
import math
from pathlib import Path
from  openvino.inference_engine import IENetwork, IEPlugin


ITER_NUM=10

INPUT_DIR = Path().cwd() / 'data'


images_paths = list(INPUT_DIR.glob('**/*.jpg'))
print('Total %d images' % len(images_paths))

MAX_IMAGES = int(math.log(max_images,2))

images = [cv2.imread(str(img_path)) for img_path in images_paths[:MAX_IMAGES]]

print('Proceed with %d images' % len(images))


# Path to an .xml file with a trained model
MODEL_PATH_XML = Path('object_detector_16-classes_frozen_graph.xml')
MODEL_PATH_BIN = MODEL_PATH_XML.stem + '.bin'
# Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. 
# Demo will look for a suitable plugin for device specified (CPU by default)
DEVICE = 'CPU'

OPENVINO_HOME = Path('C:\\Intel\\computer_vision_sdk_2018.4.420')
CPU_AVX_EXTENSION = OPENVINO_HOME / 'deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll'


# images - arrays of np_arrays
def get_openvino_results(frame, exec_net, out_blob, cur_request_id):
    if exec_net.requests[cur_request_id].wait(-1) == 0:
        #inf_end = time.time()
        #det_time = inf_end - inf_start

        # Parse detection results of the current request
        res = exec_net.requests[cur_request_id].outputs[out_blob]
        return res[0][0]
        
        # Draw performance stats
        #inf_time_message = "Inference time: N\A for async mode" if is_async_mode else "Inference time: {:.3f} ms".format(det_time * 1000)
        #render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
        #async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else "Async mode is off. Processing request {}".format(cur_request_id)
    else:
        return None

        
def test_openvino_inference(images, exec_net, input_blob, out_blob, input_shape, is_async_mode=True):
    results = []
    
    n, c, h, w = input_shape
    
    cur_request_id = 0
    next_request_id = 1
            
    frame = None
    next_frame = None
        
    i = 0
    for img in images:
        if is_async_mode:
            next_frame = img
        else:
            frame = img

        initial_h, initial_w = img.shape[:2]
        
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        #inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
            
        if frame is not None:
            results.append(get_openvino_results(frame, exec_net, out_blob, cur_request_id))
            
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
    
    results.append(get_openvino_results(frame, exec_net, out_blob, cur_request_id))
        
    return results
    
            
def test_openvino(images, max_images=1, iter_num=1):
    model_xml = str(MODEL_PATH_XML)
    model_bin = str(MODEL_PATH_BIN)

    plugin = IEPlugin(device=DEVICE, plugin_dirs=None)

    if CPU_AVX_EXTENSION and DEVICE is 'CPU':
        plugin.add_cpu_extension(str(CPU_AVX_EXTENSION))

    #print("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)

    if DEVICE == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("Following layers are not supported by the plugin for specified device {}:\n {}".format(plugin.device, ', '.join(not_supported_layers)))
            print("Please try to specify cpu extensions library path in demo's command line parameters using -l or --cpu_extension command line argument")
            return


    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    #print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)


    # Read and pre-process input image
    input_shape = net.inputs[input_blob].shape
    del net

    results = None

    for n in [1 << i for i in range(int(math.log(max_images,2))+1)]:
        sum_time = 0
        for i in range(iter_num):
            all_inf_start = time.time()
            results = test_openvino_inference(images[:n], exec_net, input_blob, out_blob, input_shape, is_async_mode=True)
            all_inf_end = time.time()
            test_time = all_inf_end - all_inf_start
            sum_time += test_time

        print('test_openvino: %d images: %.3f ms' % (n, (sum_time/iter_num)*1000))

    del exec_net
    del plugin

    results_out = []
    for res in results:
        img_objects = []
        for obj in res:
            img_objects.append({
                'score': round(obj[2],2),
                'cls': obj[1],
                'xmin': round(obj[3],2),
                'ymin': round(obj[4],2),
                'xmax': round(obj[5],2),
                'ymax': round(obj[6],2),
            })

        results_out.append(img_objects)

    assert len(results_out)==n, '%d != %d' % (len(results_out), n)
        
    return results_out

openvino_results = test_openvino(images, max_images=MAX_IMAGES, iter_num=ITER_NUM)
