from Controller.TFL_manager import TFLManager
import pickle


class Controller:
    def __init__(self, frames_file_path):
        self.frames_file_path = frames_file_path


    def find_id_frame(self, path):
        return int(path.split('_')[2])


    def run(self):
        with open(self.frames_file_path, "r+") as frames_file:
            pathes_list = frames_file.readlines()
        
        pathes_list = [path[:-1] for path in pathes_list]
        self.frames_pathes_list = pathes_list[1:]

        with open(pathes_list[0], 'rb') as data_file:
            data = pickle.load(data_file, encoding='latin1')

        self.TFL_manager = TFLManager(data)

        for path in self.frames_pathes_list:
            id_frame = self.find_id_frame(path)
            self.TFL_manager.run(id_frame, path)
