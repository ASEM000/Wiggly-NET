
import cv2
from numba import jit ,int8 , prange ,f8 ,i8 
import numpy as np

@jit(nopython=True)
def streak(row,tolerance=0):
    '''
    Input 
    *Row       : numpy array of binary pixels in shape (frame,row,col,channel) = (1,1,col,channel)
    *Tolerance : pixel tolerance in counting the streak ( tolerance = 0 means strictly next pixel ) 

    Output
    *same size row with zeros except for the longest streak
    '''
    locs = np.where( row == 255 )[0]  
    if len(locs) == 0 :return row 
    ptr=0 ; streak_locs = np.array([locs[0]])   
    
    for i in range(1,len(locs)):
        if locs[i]-locs[i-1] > ( tolerance + 1) :# take action if the streak is discontinued
            if (i)-ptr >= len(streak_locs) :streak_locs = locs[ptr:i] #check if the discontinued streak[ptr,i) is larger or equal then replace the existing one
            ptr = i #start over
        elif i == (len(locs) -1) :# if this is the last element ; and no streak discontinued
            if (i)-ptr >= len(streak_locs) : streak_locs = locs[ptr:]
                 
    row[:] = 0 ; row[streak_locs] = 255
    return row

@jit(nopython=True,parallel=True)
def denoise(array,tolerance=0):
    '''
    Input : array in shape of (frame,row,col,1)
    Output: array of shape of (frame,row,col,1)
    '''
    frame,row,col,channel = array.shape
    result  = np.zeros((frame,row,col,1),dtype=np.uint8)
    for fi in prange(frame):
        for ri in prange(row) :result[fi,ri]=streak(array[fi,ri],tolerance=tolerance)
    return result    

def threshold(array,value):
    '''
    Input : array in shape of (frame,row,col,channel)
    Output: array of shape of (frame,row,col,1)
    '''
    frame,row,col,channel = array.shape
    result= np.zeros((frame,row,col,1),dtype=np.uint8)
    for fi in range(frame):
        gray=cv2.cvtColor(array[fi],cv2.COLOR_BGR2GRAY)
        _,result[fi,:,:,0]=cv2.threshold(gray,value,255,cv2.THRESH_BINARY_INV);
    return result


def denoise_pipeline(array , threshold_value , tolerance):
    '''
    Input  : 4D input numpy array ( frame,row,col,channel)
    Output : 2D Jet center array (row , frame)  for each row track the center of the jet
    '''
    thresh_video = threshold(array ,value = threshold_value)
    denoised_video =denoise(thresh_video ,tolerance = tolerance) 
    return denoised_video
