
import http.client, urllib.parse, json
from os.path import expanduser
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='8'
from tqdm import tqdm

subscription_key = '...'
uri_base = '...'


headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

params = urllib.parse.urlencode({
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
})

#Call Face API
imgs_list = []
for k in tqdm(range(0,1)):
    img = open(expanduser(f'./{str(k).zfill(4)}.jpeg'), 'rb')
    conn = http.client.HTTPSConnection('...')
    conn.request("POST", "/face/v1.0/detect?%s" % params, img, headers)
    response = conn.getresponse()
    data = response.read()
    face_dict = json.loads(data)
    attri_list = []
    attri_list.append(face_dict[0]['faceAttributes']['age'])
    attri_list.append(face_dict[0]['faceAttributes']['facialHair']['beard'])
    if face_dict[0]['faceAttributes']['gender']=='male':
        attri_list.append(0)
    else:
        attri_list.append(1)
    if face_dict[0]['faceAttributes']['glasses']=='NoGlasses':
        attri_list.append(0)
    elif face_dict[0]['faceAttributes']['glasses']=='ReadingGlasses':
        attri_list.append(1)
    elif face_dict[0]['faceAttributes']['glasses']=='Sunglasses':
        attri_list.append(2)
    else:
        attri_list.append(3)
    attri_list.append(face_dict[0]['faceAttributes']['hair']['bald'])
    attri_list.append(face_dict[0]['faceAttributes']['smile'])
    conn.close()
    imgs_list.append(attri_list)

imgs_array = np.array(imgs_list)

np.save(f'./attributes_woman.npy',imgs_array)
attributes = np.load(f'./attributes_woman.npy')
print('attributes',attributes.shape)


    