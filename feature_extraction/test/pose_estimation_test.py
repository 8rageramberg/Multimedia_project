import sys
import numpy as np
sys.path.append("feature_extraction")
from extractors.pose_estimation import pose_estimation


def compare_masks(visualized_mask1, segmentation_mask1, visualized_mask2, segmentation_mask2):
    visualized_diff = np.abs(visualized_mask1 - visualized_mask2)
    segmentation_diff = np.abs(segmentation_mask1 - segmentation_mask2)
    
    visualized_diff_sum = np.sum(visualized_diff)
    segmentation_diff_sum = np.sum(segmentation_diff)
    
    return visualized_diff_sum, segmentation_diff_sum

test_1 = 'archive/hip thrust/hip thrust_100081.jpg'     # Hard photo, it doesnt work very good here. it may be somehting about turning the photo upside down
test_2 = 'archive/shoulder press/shoulder press_100041.jpg'

test_1 = 'archive/A_test_set/Test_plank/plank_g3_copy.jpg'
test_2 = 'archive/A_test_set/Test_plank/plank_704601 copy.jpg'

test_1 = pose_estimation(test_1)
vis_1, seg_1 = test_1.get_features(display=True) 


# Flatten the masks

flat_visualized_mask = vis_1.reshape(-1, 3)  # Reshape to 2D for PCA
flat_segmentation_mask = seg_1.flatten()

test_2 = pose_estimation(test_2)
vis_2, seg_2 = test_2.get_features(display=True )

visualized_diff, segmentation_diff = compare_masks(vis_1, seg_1, vis_2, seg_2)

print("Visualized mask difference:", visualized_diff/1000)
print("Segmentation mask difference:", segmentation_diff/1000)