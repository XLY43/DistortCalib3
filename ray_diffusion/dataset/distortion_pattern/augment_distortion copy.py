import numpy as np
import skimage
import skimage.io as io
from skimage.transform import rescale
import scipy.io as scio
import argparse
import os

from ray_diffusion.dataset.distortion_pattern.distortion_model import distortionModel, distortionParameter


def augment_distortion(types, image):
    # image [H, W, 3]    
    width  = image.shape[1]
    height = image.shape[0]

    parameters = distortionParameter(types)

    disImg = np.array(np.zeros(image.shape), dtype = np.uint8)
    u = np.array(np.zeros((image.shape[0],image.shape[1])), dtype = np.float32)
    v = np.array(np.zeros((image.shape[0],image.shape[1])), dtype = np.float32)

    # # crop range
    # xmin = int(width*1/4)
    # xmax = int(width*3/4 - 1)
    # ymin = int(height*1/4)
    # ymax = int(height*3/4 - 1)

    for i in range(width):
        for j in range(height):
            
            xu, yu = distortionModel(types, i, j, width, height, parameters)
            
            if (0 <= xu < width - 1) and (0 <= yu < height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j
                
                # Bilinear interpolation
                Q11 = image[int(yu), int(xu), :]
                Q12 = image[int(yu), int(xu) + 1, :]
                Q21 = image[int(yu) + 1, int(xu), :]
                Q22 = image[int(yu) + 1, int(xu) + 1, :]
                
                disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                 Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                 Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                 Q22*(xu - int(xu))*(yu - int(yu))
                            
    u = np.expand_dims(u, axis=2)
    v = np.expand_dims(v, axis=2)
    flow = np.concatenate((u, v), axis=2)
    return disImg, flow

def _crop_flow(flow, bbox):
    u = flow[:,:,0]
    v = flow[:,:,1]
    # flow [H, W, 2]    
    width  = flow.shape[1]
    height = flow.shape[0]

    crop_u  = np.array(np.zeros((int(height/2),int(width/2))), dtype = np.float32)
    crop_v  = np.array(np.zeros((int(height/2),int(width/2))), dtype = np.float32)

    # top: bbox[1]
    # left: bbox[0]
    # height: bbox[3] - bbox[1]
    # width : bbox[2] - bbox[0]
    ymin = bbox[1]
    ymax = bbox[3]
    xmin = bbox[0]
    xmax = bbox[2]
    # # crop range
    # xmin = int(width*1/4)
    # xmax = int(width*3/4 - 1)
    # ymin = int(height*1/4)
    # ymax = int(height*3/4 - 1)
    for i in range(width):
        for j in range(height):
            if(xmin <= i <= xmax) and (ymin <= j <= ymax):
                crop_u[j - ymin, i - xmin] = u[j,i]
                crop_v[j - ymin, i - xmin] = v[j,i]
    crop_u = np.expand_dims(crop_u, axis=2)
    crop_v = np.expand_dims(crop_v, axis=2)
    flow = np.concatenate((crop_u, crop_v), axis=2)
    return flow

def generatepindata(types, image):
    # image [H, W, 3]    
    width  = image.shape[1] //2
    height = image.shape[0] //2

    parameters = distortionParameter(types)
    
    OriImg = image
    temImg = rescale(OriImg, 0.5, mode='reflect', channel_axis=2)
    ScaImg = skimage.img_as_ubyte(temImg)

    padImg = np.array(np.zeros((ScaImg.shape[0]+1,ScaImg.shape[1]+1, 3)), dtype = np.uint8)
    padImg[0:height, 0:width, :] = ScaImg[0:height, 0:width, :]
    padImg[height, 0:width, :] = ScaImg[height - 1, 0:width, :]
    padImg[0:height, width, :] = ScaImg[0:height, width - 1, :]
    padImg[height, width, :] = ScaImg[height - 1, width - 1, :]

    disImg = np.array(np.zeros(ScaImg.shape), dtype = np.uint8)
    u = np.array(np.zeros((ScaImg.shape[0], ScaImg.shape[1])), dtype = np.float32)
    v = np.array(np.zeros((ScaImg.shape[0],ScaImg.shape[1])), dtype = np.float32)

    for i in range(width):
        for j in range(height):
            
            xu, yu = distortionModel(types, i, j, width, height, parameters)
            
            if (0 <= xu <= width - 1) and (0 <= yu <= height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j
                
                # Bilinear interpolation
                Q11 = padImg[int(yu), int(xu), :]
                Q12 = padImg[int(yu), int(xu) + 1, :]
                Q21 = padImg[int(yu) + 1, int(xu), :]
                Q22 = padImg[int(yu) + 1, int(xu) + 1, :]
                
                disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                 Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                 Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                 Q22*(xu - int(xu))*(yu - int(yu))
    

    u = np.expand_dims(u, axis=2)
    v = np.expand_dims(v, axis=2)
    flow = np.concatenate((u, v), axis=2)
    return disImg, flow