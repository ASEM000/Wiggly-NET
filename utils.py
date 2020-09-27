import numpy as np
from sklearn.model_selection import train_test_split
import sys

def draw_bbox(array,bbox,t=3,color=(255,0,255)):
    '''
    Input : 3D array (row,col,channel)
    Output: 3D array (row,col,channel) with bounding box
    '''
    x,y,w,h = bbox
    array[y:y+t,x:x+t] = color         #draw cross
    
    x = x - w//2
    y = y - h//2
    
    h=h-t
    w=w-t
    
    array[y:y+h+1,x:x+t] = color       #left vertical line
    array[y:y+h+1,x+w:x+w+t] = color   #right vertical line
    array[y:y+t,x:x+w] = color         #top horizontal line
    array[y+h:y+h+t,x:x+w+t] = color   #bottom horizontal line
    
    return array


def crop_bbox(array,bbox):
    '''
    x and y are center of the bbox
    
    Input : 3D array (row,col,channel)
    Output: 3D array (row,col,channel) cropped
    '''
    x,y,w,h = bbox
    

    
    x = x - w//2
    y = y - h//2
    
    if len(array.shape)==3:return array[y:y+h,x:x+w]#single color image
    elif len(array.shape)==4:return array[:,y:y+h,x:x+w]#video (frame,row,col,channel)



def draw_hline(array,pos=0,t=1,color=(255,0,255)):
    array[pos-t:pos+t,:,:] = color ;
    return array


def convert_video_to_train_test(array,l,df,test_size=0.05,random_state=42,verbose=False):
    '''
    Input : 4D numpy array (frame,row,col,channel)
    Output: 5D numpy array (sample,frame,row,col,channel)
    
    extract frame sequence defined by sequence length (l) and frame separation (df)
    
    **For example 
    l = 3 ,
    df = 5 
    will produce a sample of n,n+5,n+10 frames array
    
    '''
    frame,row,col,channel = array.shape
    sample_size = (frame) - df*l
    result = np.zeros((sample_size,l,row,col,channel),dtype=np.float32)
    
    if verbose : 
        print(f'the reuslt size :{sys.getsizeof(result)/(1024**2)} MB')
        print(f'the result shape:{result.shape}')
    
    for si in range(sample_size):
        result[si,:,:,:,:] = array[si:si+df*l:df,:,:,:] / 255.0   #divide by 255 for better convergence
        
    x,xx,y,yy = train_test_split(result[:,:-1,:,:,:],
                                 result[:,-1:,:,:,:],
                                 test_size=test_size, 
                                 random_state=random_state)
        
    return x,xx,y,yy