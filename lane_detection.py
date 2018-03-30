import cv2
import numpy as np
import math


def getMidPoint(p1,p2):
    return (p1[0]+p2[0])/2,(p1[1]+p2[1])/2


def findintersection(p1,p2,p3,p4):#function to find intersection of two line
    slope_m1 = (p2[1]-p1[1])/float(p2[0]-p1[0])
    slope_m2 = (p4[1]-p3[1])/float(p4[0]-p3[0])
    c1 = -slope_m1*p2[0] + p2[1]
    c2 = -slope_m2*p4[0] + p4[1]
    
    if(slope_m1!=slope_m2):
        x = (c1-c2)/float(slope_m2-slope_m1)
        y = slope_m1*x + c1
    else:
        return -1,-1
    return int(x),int(y)


def region_of_interest(edges,vertices): #will remove the unwanted portion of image
    vertices = np.array([vertices],dtype=np.int32)
    mask = np.zeros(edges.shape,np.uint8)
    cv2.fillPoly(mask,vertices,(255,255,255))
    masked_image=cv2.bitwise_and(mask,edges)
    return masked_image



def hough_lines(roi):#drawing hough line and approximating lane line
    rho=1
    theta = np.pi/180
    threshold=40
    minLineLength = 30
    maxLineGap=200
    lines = cv2.HoughLinesP(roi,rho,theta,threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)
    
    ''' 
    print len(lines[0][0])
    print lines[0][0]
    cv2.waitKey(0)'''
    lines_img = np.zeros(roi.shape,np.uint8)
    left_lines = []
    right_lines =[]
    maxleft =0
    maxright=0
    
    for line in lines[0]:
        p1 = (line[0],line[1])
        p2 = (line[2],line[3])
        slope = (line[3]-line[1])/float(line[2]-line[0])
        if slope >=.3:
           left_lines.append(line)
        elif slope<-0.3:
            right_lines.append(line)
    for line in left_lines:
        length = math.hypot(line[0]-line[2],line[1]-line[3])
        if length > maxleft:
            maxleft = length
            left_lane= line
    for line in right_lines:
        length = math.hypot(line[0]-line[2],line[1]-line[3])
        if length > maxright:
            maxleft = length
            right_lane= line

    cv2.line(lines_img , (left_lane[0],left_lane[1]),(left_lane[2],left_lane[3]),(255,213,234),3)
    
    cv2.line(lines_img , (right_lane[0],right_lane[1]),(right_lane[2],right_lane[3]),(255,213,234),3)        
    return lines_img , (left_lane,right_lane)        
        
        
    
        

for index in range(218):
    
    filepath = 'video1_captures/capture'+str(index) + '.jpg'
    img = cv2.imread(filepath)
    imgHeight,imgWidth,_=img.shape
    blur = cv2.GaussianBlur(img,(5,5),0)
    edges = cv2.Canny(blur,0,50)
    vertices = ((3*imgWidth/5,3*imgHeight/5),(imgWidth/4,3*imgHeight/5),(40,imgHeight),(imgWidth-40,imgHeight))
    roi = region_of_interest(edges,vertices)
    linesImage,lane=hough_lines(roi)
    llane = lane[0]
    rlane = lane[1]
    cv2.imshow('achu',linesImage)
    refline = [(0,int(0.8*imgHeight)),(imgWidth,int(0.8*imgHeight))]
    cv2.line(linesImage,refline[0],refline[1],(255,122,123),1)
    p3 = findintersection((llane[0],llane[1]),(llane[2],llane[3]),refline[0],refline[1])
    p4 = findintersection((rlane[0],rlane[1]),(rlane[2],rlane[3]),refline[0],refline[1])
    cl = getMidPoint(p3,p4)
    cc = getMidPoint(refline[0],refline[1])
    cv2.circle(img,cl,10,(255,0,0),-1)
    cv2.circle(img,cc,8,(45,145,89),-1)
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold,minlinelenghth,maxlinegap)
    cv2.imshow('region of interest',roi)
    cv2.imshow('output',img)
    #cv2.imshow('edges',edges)
    cv2.imshow('as',linesImage)
    cv2.waitKey(10)
    index=index+1

cv2.destroyAllWindows()    

