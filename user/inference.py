import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided. 

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################

        # The following is a dummy cell detection algorithm
        try:    
            model = tf.keras.models.load_model('ocelot400.h5')
            result = self.one_image_predictions(model, cell_patch)
            result_inv = (255-result).astype(np.uint8)
            ret, th = cv2.threshold(result_inv, 47, 255, cv2.THRESH_BINARY)
            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
            dilation = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))
            contours_res, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours_res:
                center_points = []
                for contour in contours_res:
                    M = cv2.moments(contour)
                    
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        center_points.append((center_x, center_y, 2, 1.0))
                return center_points
            else :
                return []
        except:
            return []
        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        
    

    def HistEqual(self, image):
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cb, cr = cv2.split(ycbcr)
        hist = cv2.equalizeHist(y)
        merged = cv2.merge([hist, cb, cr])
        res = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        return res


    def crop_image(image, patch_size):
        image_height, image_width, channels = image.shape
        patch_height, patch_width = patch_size

        num_patches_vertical = image_height // patch_height
        num_patches_horizontal = image_width // patch_width

        patches = image[:num_patches_vertical * patch_height, :num_patches_horizontal * patch_width, :]
        patches = np.split(patches, num_patches_vertical, axis=0)
        patches = [np.split(row_patches, num_patches_horizontal, axis=1) for row_patches in patches]
        patches = np.array(patches).reshape(-1, patch_height, patch_width, channels)

        return patches


    def parse_image(self, image_path):
        image_equal = self.HistEqual(image_path)
        image_equal = cv2.cvtColor(image_equal, cv2.COLOR_BGR2RGB)
        image_equal = image_equal/255
        result =self.crop_image(image_equal, (128, 128))
        return result

    def concat_patch(image_pred_cont):
        num_patches = len(image_pred_cont)

        num_patches_per_row = int(np.sqrt(num_patches))
        num_patches_per_col = num_patches // num_patches_per_row

        concatenated_image = np.concatenate([np.concatenate(image_pred_cont[i*num_patches_per_col:(i+1)*num_patches_per_col], axis=1) for i in range(num_patches_per_row)], axis=0)

        return concatenated_image
    
    '''
    Make a code to receive 1 image with size 1024 x 1024
    make it into 128 x 128 so it will be 16 patch images
    detection using a pretrained model
    make 16 patch images into 1 image 1024 x 1024
    Show it
    '''

    def one_image_predictions(self,pretrained_model, image_path, mask_path=None):
        image_patch = self.parse_image(image_path)
        image_pred_cont = []
        for one_patch_image in image_patch:
            temp = np.expand_dims(one_patch_image, axis=0)
            predictions = pretrained_model.predict(temp)
            # preds_sigmoid = keras.activations.sigmoid(predictions)
            predicted_result = np.squeeze(predictions, axis=0)
            gray_image = predicted_result[:, :, 0]
            normalized_image = (gray_image * 255).astype(np.uint8)
            image_pred_cont.append(normalized_image)
        result = self.concat_patch(image_pred_cont)
        return result
