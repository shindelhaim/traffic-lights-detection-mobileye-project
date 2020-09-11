from Model.frame_container import FrameContainer
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def marking_coordinates(r_image, coordinates, auxiliary):  
    for i in range(len(coordinates)):
        if auxiliary[i] == 'R':
            color = [255,0,0]
    
        elif auxiliary[i] == 'G':
            color = [0,255,0]

        r_image[coordinates[i][0]: coordinates[i][0] + 8 , coordinates[i][1]: coordinates[i][1] + 8] = [[color,color,color,color,color,color,color,color]
                                                                                                            ,[color,color,color,color,color,color,color,color]
                                                                                                            ,[color,color,color,color,color,color,color,color]
                                                                                                            ,[color,color,color,color,color,color,color,color]
                                                                                                            ,[color,color,color,color,color,color,color,color]
                                                                                                            ,[color,color,color,color,color,color,color,color]
                                                                                                            ,[color,color,color,color,color,color,color,color]
                                                                                                            ,[color,color,color,color,color,color,color,color]]
                                            
    return r_image


def display_results_visually(r_image, current_frame, data_candidates, distances):
        fig, (ax_can, ax_trl, ax_dist) = plt.subplots(3, 1, figsize=(12, 30))
        r_image1 = np.copy(r_image)

        img = marking_coordinates(r_image[:,:,:], data_candidates[0], data_candidates[1])
        ax_can.imshow(img, cmap='gray')
        ax_can.set_ylabel('Candidates')
        
        img1 = marking_coordinates(r_image1, current_frame.traffic_light, current_frame.auxiliary)
        ax_trl.imshow(img1, cmap='gray')
        ax_trl.set_ylabel('Traffic_light')
        
        if distances != []:
            ax_dist.imshow(img1, cmap='gray')
        
            for i in range(len(current_frame.traffic_light)):
                ax_dist.text(current_frame.traffic_light[i, 1], current_frame.traffic_light[i, 0], r'{0:.1f}'.format(distances[i]), color='y')

            ax_dist.set_ylabel('Distance')

        fig.suptitle(f'Frame #{current_frame.id} {current_frame.img_name}', fontsize=16)
        plt.show()
