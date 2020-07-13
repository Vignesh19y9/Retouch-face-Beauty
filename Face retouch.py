import dlib
import cv2
import numpy as np

vis=True
smooth=31
glowv=3
tonev=0
text=3
screenv=0
image = cv2.imread('input1.jpg')
width=450
image=cv2.resize(image,(width,int(width*image.shape[0]/image.shape[1])))


def alphaBlend(img1, img2, mask):
    
    mask=~mask
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    
    return blended

def overlay(img1,img2):
      
    a = img1
    b = img2
    a = a.astype(float)/255
    b = b.astype(float)/255    
    mask = a >= 0.5 # generate boolean mask of everywhere a > 0.5 
    ab = np.zeros_like(a) # generate an output container for the blended image 
    # now do the blending 
    ab[~mask] = (2*a*b)[~mask] # 2ab everywhere a<0.5
    ab[mask] = (1-2*(1-a)*(1-b))[mask]
    # else this 
    out=(ab*255).astype(np.uint8) 
    
    return out

def retouch(img):
       
# -----------------------  Smoothing -------------
    invert=~img
    blur = cv2.GaussianBlur(invert,(smooth,smooth),0)
    filtered = invert - blur
    filtered = filtered + 127*np.ones(filtered.shape, np.uint8)   
    out=overlay(img,filtered)
#--------------------------- Glow -------------
    glow=out
    glow = cv2.GaussianBlur(glow,(glowv,glowv),0)
    
    a = glow.astype(float)/255    
    d=1-(1-a)*(1-a)
    screen=(d*255).astype(np.uint8) 
    screenMask=np.ones((glow.shape[0],glow.shape[1]),dtype='uint8')*screenv
    screen=alphaBlend(screen,glow,screenMask)
#--------------------------- Tone ---------------
    if tonev<0:
        c=0
    else:c=2
    tone = np.int16(screen)
    tone[:,:,c] = tone[:,:,c] * (tonev/127+1) - tonev + tonev
    tone = np.clip(tone, 0, 255)
    tone = np.uint8(tone)
    screen=tone    
    
#--------------------------------- Texture
    
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    blur = cv2.GaussianBlur(gray,(text,text),0)   
    filtered = gray - blur
    filtered = filtered + 127*np.ones(filtered.shape, np.uint8)
    out=overlay(screen,filtered)

    
    if vis==False:    
        cv2.imshow("water", out)
        
        cv2.waitKey(1)
    return out




def shape_to_np(shape, dtype="int"):
    
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


FACIAL_LANDMARKS_IDXS = {"mouth": (48, 68),"inner_mouth": (60, 68),"right_eyebrow": (17, 22),"left_eyebrow": (22, 27),"right_eye": (36, 42),"left_eye": (42, 48),"nose": (27, 36),"jaw": (0, 17)}


gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
visIm=image.copy()

detector = dlib.get_frontal_face_detector()
predictor68 = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 
predictor81 = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat') 

faces = detector(gray, 1)
print("No of face ",len(faces))


landmarks = predictor68(gray, faces[0])#best for bottom face
landmarks = shape_to_np(landmarks)
landmarks81 = predictor81(gray, faces[0])#best for top face
landmarks81 = shape_to_np(landmarks81)

landmarksf = []
for l in range(len(landmarks81)):
    
    if l>= 68:
        print(landmarks81[l])
        landmarksf.append(landmarks81[l])
    else:
        landmarksf.append(landmarks[l])
        
landmarksf=np.array(landmarksf)#combine


## =============================================================================
## bottomface
## =============================================================================
#bottomFace=landmarks[0:17]
#bottomFaceMask = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
#hull = cv2.convexHull(bottomFace)
#cv2.drawContours(bottomFaceMask, [hull], -1, (255, 255, 255), -1)

# =============================================================================
# Full face
# =============================================================================
fullFaceMask = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')

hull = cv2.convexHull(landmarksf)
cv2.drawContours(fullFaceMask, [hull], -1, (255, 255, 255), -1)


if vis==True:
    cv2.drawContours(visIm, [hull], -1, (0, 0, 255), 2)
    

# =============================================================================
# Kmeans
# =============================================================================
x = cv2.bitwise_and(image, image, mask = fullFaceMask)
Z = x.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
label_list=list(label)

if label_list.count(0)>label_list.count(1):
    center[0]=[0,0,0]
    center[1]=[255,255,255]
else:
    center[0]=[255,255,255]
    center[1]=[0,0,0]
    
res = center[label.flatten()]
res = res.reshape((image.shape))    
Kmask=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

## =============================================================================
## top face
# =============================================================================
topFace=landmarks81[68:81]
topFaceMask = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
hull = cv2.convexHull(topFace)

cv2.drawContours(topFaceMask, [hull], -1, (255, 255, 255), -1)

if vis==True:
    cv2.drawContours(visIm, [hull], -1, (0, 0, 255), 2)
    
# =============================================================================
# eyebrow
# =============================================================================
eyeBrowMask = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')

eyeBrow=[(17, 22), (22, 27)]

for val in eyeBrow:
    
    pts=landmarks[val[0]:val[1]]
    hull = cv2.convexHull(pts)
    cv2.drawContours(eyeBrowMask, [hull], -1, (255, 255, 255), -1)

    if vis==True:
        cv2.drawContours(visIm, [hull], -1, (0, 0, 255), 2)
# =============================================================================
# eye
# =============================================================================
eyeMask = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')

eye=[(36, 42),(42, 48)]

for val in eye:
    
    pts=landmarks[val[0]:val[1]]
    hull = cv2.convexHull(pts)
    cv2.drawContours(eyeMask, [hull], -1, (255, 255, 255), -1)

    if vis==True:
        cv2.drawContours(visIm, [hull], -1, (0, 0, 255), 2)
        
#================================================================
# Lips
# =============================================================================
lipMask = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
lip=landmarks[48:68]

hull = cv2.convexHull(lip)
cv2.drawContours(lipMask, [hull], -1, (255, 255, 255), -1)

#lipMask = cv2.dilate(lipMask, kernel,iterations=1) 
if vis==True:
    cv2.drawContours(visIm, [hull], -1, (0, 0, 255), 2)
    
# =============================================================================
# Nose    
# =============================================================================
    
noseMask = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
nose=landmarks[31:36]
hull = cv2.convexHull(nose)
cv2.drawContours(noseMask, [hull], -1, (255, 255, 255), -1)
if vis==True:
    cv2.drawContours(vis, [hull], -1, (0, 0, 255), 2)
# =============================================================================
# 
# =============================================================================
noseMask2 = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
noseeoEyeLeft=[tuple(landmarks[31]),tuple(landmarks[39])]
noseeoEyeRight=[tuple(landmarks[35]),tuple(landmarks[42])]

cv2.line(noseMask2,noseeoEyeLeft[0],noseeoEyeLeft[1], (255, 255, 255), 10) 
cv2.line(noseMask2,noseeoEyeRight[0],noseeoEyeRight[1], (255, 255, 255), 10) 
if vis==True: 
    cv2.line(vis,noseeoEyeLeft[0],noseeoEyeLeft[1], (255, 255, 255), 5) 
    cv2.line(vis,noseeoEyeRight[0],noseeoEyeRight[1], (255, 255, 255), 5)  


# =============================================================================
# 
# =============================================================================
topFaceMask2 = cv2.bitwise_and(topFaceMask, Kmask, mask = None)
topFaceMask2=cv2.blur(topFaceMask2,(2,90))
Kmask2=topFaceMask2
Kmask2[Kmask==255]=255
finalMask=Kmask2
finalMask[fullFaceMask==255]=100
finalMask[Kmask==255]=255

kernel = np.ones((5,5), np.uint8) 
eyeMask1 = cv2.dilate(eyeMask, kernel,iterations=2) 
lipMask1 = cv2.dilate(lipMask, kernel,iterations=2) 

finalMask[eyeBrowMask==255]=0
finalMask[eyeMask1==255]=0
finalMask[lipMask1==255]=0
finalMask[noseMask==255]=0
finalMask[noseMask2==255]=120
finalMask=cv2.blur(finalMask,(20,20))

finalMask[eyeBrowMask==255]=0
finalMask[eyeMask1==255]=0
finalMask[lipMask1==255]=0
finalMask[noseMask==255]=0
finalMask[noseMask2==255]=100
finalMask[topFaceMask2==0]=0
finalMask=cv2.blur(finalMask,(5,5))

finalMask[fullFaceMask==0]=0
# =============================================================================
# 
# =============================================================================
segmented0 = alphaBlend(image, image,  finalMask)
retouchi=retouch(image)
segmented = alphaBlend(retouchi, image,  finalMask)
#cv2.imwrite('d11.jpg',segmented)
# =============================================================================
# 
# =============================================================================
if vis == True:
    cv2.imshow("vis", visIm)
    cv2.imshow("fullFaceMask", fullFaceMask)
    cv2.imshow("topFaceMask", topFaceMask)
    cv2.imshow("topFaceMask2", topFaceMask2)
    cv2.imshow("Kmask", Kmask)
    cv2.imshow("Kmask2", Kmask2)
    cv2.imshow("finalMask", finalMask)
    
cv2.imshow("image", image)
cv2.imshow("segmented", segmented)   
cv2.waitKey(0)
