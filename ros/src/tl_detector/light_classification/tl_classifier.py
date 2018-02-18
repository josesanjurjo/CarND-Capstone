from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import cv2
import rospy
import os.path


model_file   = 'ssd_mobilenet_v1_coco_11_06_2017_frozen_inference_graph.pb'

TL_CLASS          = 10

class TLClassifier(object):
    def __init__(self):
        try:
           model = '{}/{}'.format(os.path.dirname (os.path.realpath(__file__)), model_file)
           
           tl_graph = tf.Graph()
           
           with tl_graph.as_default():
                graph_def = tf.GraphDef()
                with tf.gfile.GFile (model, 'rb') as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(graph_def, name='')
                    
           self._image_tensor      = tl_graph.get_tensor_by_name('image_tensor:0') 
           self._detection_boxes   = tl_graph.get_tensor_by_name('detection_boxes:0')
           self._detection_scores  = tl_graph.get_tensor_by_name('detection_scores:0')
           self._detection_classes = tl_graph.get_tensor_by_name('detection_classes:0') 

           self._TF_session = tf.Session(graph=tl_graph)

        except Exception as e:
            rospy.logfatal (e)
            
    def get_bboxes (self, image, min_score):
        """
        Return all the bounding boxes of traffic lights where the detection score is higher than a threshold given as an input.
        """
        # Run Tensorflow and get the bounding boxes, the classes and the detection scores
        tensor = image [np.newaxis]
        (boxes, scores, classes) = self._TF_session.run(
            fetches   = [self._detection_boxes, self._detection_scores, self._detection_classes],
            feed_dict = {self._image_tensor: tensor}
        )

        # convert the relative coordinates to image space
        boxes *= [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]
        
        # remove the leading dimension of 1 (TF required a 4D tensor), so that the problem "index 1 is out of bounds for axis 0 with size 1" does not appear
        boxes   =   boxes.squeeze().astype(int)
        classes = classes.squeeze().astype(int)
        scores  =  scores.squeeze()

        # filter the boxes with detection score lower than the minimum set
        filtered = np.argwhere((classes == TL_CLASS) & (scores > min_score)).flatten()

        return boxes[filtered]
        
    def classify_bbox (self, image, bbox):
        """Classify a bounding bbox in the image
        """
        result = TrafficLight.UNKNOWN

        y0, x0, y1, x1 = bbox
        tl = image[y0:y1, x0:x1]
        
        # Separate red yellow green and rest in the HSV space
        hue_threshold = 0, 15, 45, 75
        # minimum accepted ratio of detection area  
	area_threshold = 0.01  

	hsv = cv2.cvtColor (tl, cv2.COLOR_RGB2HSV)
	mask = np.zeros_like(hsv)

	# set upper and lower boundaries of Hue, Saturation, and Value
	for i in range(3):
		lower = np.array ([ hue_threshold[i  ],   0, 200 ])
		upper = np.array ([ hue_threshold[i+1], 255, 255 ])

		mask [..., i] = cv2.inRange (hsv, lower, upper)

	# calculate area ratio, and check if it's above the set threshold
        area_ratio = np.array ([
	     np.count_nonzero(mask[...,0]),
	     np.count_nonzero(mask[...,1]),
	     np.count_nonzero(mask[...,2]),
	], dtype=np.float) / np.product(mask.shape[:2])

	if (area_ratio > area_threshold).any():
	    result = area_ratio.argmax()

        light = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN, TrafficLight.UNKNOWN, TrafficLight.UNKNOWN] [result]
                
        return light



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        result = TrafficLight.UNKNOWN

        try:
            # SSD to get TL bounding boxes and clasification confidence scores
            min_score = 0.1
            tl_boxes  = self.get_bboxes(image, min_score) 

            # classify all traffic lights
            lights = np.array ([self.classify_bbox(image, bbox) for bbox in tl_boxes])

	    if (1 <= len(lights) <= 3) :
		if (lights == TrafficLight.RED).any() or (lights == TrafficLight.YELLOW).any():
			result = TrafficLight.RED
		else:
			result = lights[0]  
        except Exception as e:
            rospy.logwarn (e)
            
        return result

