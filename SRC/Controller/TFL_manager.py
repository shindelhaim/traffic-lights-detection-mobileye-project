from Model.frame_container import FrameContainer
from Controller.run_attention import find_light_sources_candidates
from View.results_visually import display_results_visually
from Controller.SFM import calc_TFL_dist
from Model.building_dataset import get_crop_img, get_padding_img
from tensorflow import keras
from PIL import Image
import numpy as np
import seaborn as sbn
import pickle


class TFLManager:
    def __init__(self, data_file):
        self.data_file = data_file
        self.principal_point = data_file['principle_point']
        self.focal_length = data_file['flx']
        self.prev_frame = None
        self.is_able_find_distances = False
        self.loaded_model = keras.models.load_model("Model\\model.h5")


    def detect_light_sources_candidates(self, current_img):
        red_x, red_y, green_x, green_y = find_light_sources_candidates(current_img)
        candidates = []
        auxiliary = []
        
        # TODO: צמצום כפילויות בצורה תקינה
        for x, y in zip(red_x, red_y):
            candidates += [[x, y]]
            auxiliary += ['R']
        
        for x, y in zip(green_x, green_y):
            if (x, y) not in candidates:
                candidates += [[x, y]]
                auxiliary += ['G']

        return np.array(candidates), np.array(auxiliary) 


    def identifies_coordinates_of_traffic_lights(self, image,current_img, candidates, auxiliary):
        padding_img = get_padding_img(image)
        traffic_light = []
        traffic_light_auxiliary = []

        for i in range(len(candidates)):
            crop_img = get_crop_img(padding_img,(candidates[i, 0] + 40, candidates[i, 1] + 40))
            predictions = self.loaded_model.predict([[crop_img]])
            
            if predictions[0][1] > 0.8:
                traffic_light += [candidates[i]]
                traffic_light_auxiliary += [auxiliary[i]]

        if len(traffic_light) > len(candidates):
            self.is_able_find_distances = False

        return np.array(traffic_light), np.array(traffic_light_auxiliary)


    def find_distance_of_traffic_lights(self, current_frame):
        curr_container = calc_TFL_dist(self.prev_frame, current_frame, self.focal_length, self.principal_point)
        distances = []
        
        if curr_container.traffic_lights_3d_location != []:
            distances = curr_container.traffic_lights_3d_location[:, 2]

        return distances

    
    def calc_EM(self, current_frame_id):
        EM = np.eye(4)

        for i in range(self.prev_frame.id, current_frame_id):
            EM = np.dot(self.data_file['egomotion_' + str(i) + '-' + str(i + 1)], EM)

        return EM


    def run(self, id_frame, img_name):
        current_frame = FrameContainer(img_name, id_frame)
        
        if self.is_able_find_distances:
            current_frame.EM = self.calc_EM(current_frame.id)

        candidates, auxiliary = self.detect_light_sources_candidates(np.array(Image.open(img_name)))
        current_frame.traffic_light, current_frame.auxiliary = self.identifies_coordinates_of_traffic_lights(np.array(Image.open(img_name)),current_frame.img, candidates, auxiliary)
        
        if self.is_able_find_distances:
            distances = self.find_distance_of_traffic_lights(current_frame)
        
        else:
            distances = []

        display_results_visually(np.array(Image.open(img_name)), current_frame, (candidates, auxiliary), distances)
        self.is_able_find_distances = True
        self.prev_frame = current_frame
