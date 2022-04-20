import io
import json
import re
import base64
import logging
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from flask import Flask, request, jsonify, abort

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)


API_KEY = "8de30d7a31a34d67bb83f652da0a3be2"
ENDPOINT = "https://computervisiontest12.cognitiveservices.azure.com/"
cv_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

API_KEY_FACE = 'b3248e9827f943fca4a9966fddbe836f'
ENDPOINT_FACE = 'https://myfaceapi2.cognitiveservices.azure.com/'
face_client = FaceClient(ENDPOINT_FACE, CognitiveServicesCredentials(API_KEY_FACE))


def ImageCaptioning(img):
    result = ""
    max_description = 3
    response = cv_client.describe_image_in_stream(img, max_description)
    for caption in response.captions:
        print('Image Description: {0}'.format(caption.text))
        result = 'I can see {0}'.format(caption.text)
        print('Confidence {0}'.format(caption.confidence * 100))
    return result


def ObjectDetection(img):
    result = ""
    detect_object = cv_client.detect_objects_in_stream(img)
    if (len(detect_object.objects) == 0):
        print("No objects detected.")
        result = "I cannot recognize any object "
        return result
    else:
        print(len(detect_object.objects))
        result = "I can recognize "
        for object in detect_object.objects:
            print("'{}' with confidence {:.2f}%".format(object.object_property, object.confidence * 100))
            result += "'{}' and ".format(object.object_property)
        result = re.sub(r'[\']', "", result)
        return result


def FaceDetection(image_file):
    response_detected_faces = face_client.face.detect_with_stream(
        image_file,
        detection_model='detection_01',
        recognition_model='recognition_04',
        return_face_attributes=['emotion', 'gender', 'age'],

    )

    print(response_detected_faces)

    if not response_detected_faces:
        raise Exception('No face detected')

    print('Number of people detected: {0}'.format(len(response_detected_faces)))

    counter = 1
    result = ""
    for face in response_detected_faces:
        gender = face.face_attributes.gender
        emotion = face.face_attributes.emotion
        age = face.face_attributes.age
        neutral = '{0:.0f}%'.format(emotion.neutral * 100)
        happiness = '{0:.0f}%'.format(emotion.happiness * 100)
        anger = '{0:.0f}%'.format(emotion.anger * 100)
        sandness = '{0:.0f}%'.format(emotion.sadness * 100)
        surprised = '{0:.0f}%'.format(emotion.surprise * 100)
        fear = '{0:.0f}%'.format(emotion.fear * 100)

        print("person " + str(counter))
        print("Gender: " + gender)
        print("Age: " + str(int(age)))
        print("neutral: " + neutral)
        print("Happy: " + happiness)
        print("Angery: " + anger)
        print("Sad: " + sandness)
        print("surprised: " + surprised)
        print("fear: " + fear)

        list = {'Neutral': emotion.neutral, 'Happy': emotion.happiness, 'Angry': emotion.anger, 'Sad': emotion.sadness,
                'Surprised': emotion.surprise, 'Fear': emotion.fear}

        final_emotion = max(list, key=list.get)
        print("Emotion of this person is " + final_emotion)
        counter = +1
        result += "\nPerson " + str(counter) + " Gender: " + gender + " Age: " + str(int(age)) + " Emotion of this person is " + final_emotion
    return result

@app.route("/AI", methods=['POST'])
def AI():
    # print(request.json)
    if not request.json or 'image' not in request.json:
        abort(400)

    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    img = io.BytesIO(img_bytes)

    choice = request.json['operation']

    if choice == 3:
        result = ImageCaptioning(img)
    elif choice == 2:
        result =ObjectDetection(img)
    elif choice == 1:
        result = FaceDetection(img)
    else:
        result = "No chosen operation"

    # access other keys of json
    # print(request.json['other_key'])

    result_dict = {'output': result}
    return result_dict
