import matplotlib.pyplot as plt


class FrameContainer:
    def __init__(self, img_path, id):
        self.id = id
        
        if img_path:
            self.img_name = img_path[img_path.rfind('\\') + 1:]
            self.img = plt.imread(img_path)
        
        self.traffic_light = []
        self.auxiliary = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
