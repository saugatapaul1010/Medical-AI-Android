from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import keras.backend as K
from datetime import datetime as dt
import numpy as np
import cv2
from cv2 import resize, INTER_AREA
import uuid
from PIL import Image
import os
import tempfile
from keras.models import load_model
import imageio
from keras.preprocessing import image


def resize_image_oct(image):
    resized_image = cv2.resize(image, (128,128)) #Resize all the images to 128X128 dimensions
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) #Convert to RGB
    return resized_image

def resize_image_pnm(image):
    resized_image = cv2.resize(image, (128,128)) #Resize all the images to 128X128 dimensions
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) #Convert to RGB
    return resized_image

app = Flask(__name__)
CORS(app)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route("/", methods = ["POST", "GET"])
def index():
  
  if request.method == "POST":
    type_ = request.form.get("type", None)

    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ",request.files)
    print("22222222222222222222222222222222222222222222222222",len(request.files))

    print(type_)
    data = None
    final_json = []
    if 'img' in request.files:
      file_ = request.files['img']
      name = os.path.join(tempfile.gettempdir(), str(uuid.uuid4().hex[:10]))
      file_.save(name)
      print("[DEBUG: %s]"%dt.now(),name)

      """
      test_image=imageio.imread(name)
      test_image=cv2.resize(test_image, (128,128), interpolation = cv2.INTER_AREA)
      test_image=np.array(test_image)
      test_image=test_image/255
      test_image=np.expand_dims(test_image, axis=0)
      """
      if(type_=="mal" or type_=='brain'):
        test_image = image.load_img(name, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        data=test_image

      elif(type_=='oct'):
        test_image = imageio.imread(name)                  #Read image using the PIL library
        test_image = resize_image_oct(test_image)          #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                  #Convert the image to numpy array
        test_image = test_image/255                        #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)    #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image

      elif(type_=='pnm' or type_=='dia_ret'):
        test_image = Image.open(name)                                  #Read image using the PIL library
        test_image = test_image.resize((128,128), Image.ANTIALIAS)     #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                              #Convert the image to numpy array
        test_image = test_image/255                                    #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)                #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image

      elif(type_=='breast'):
        test_image = Image.open(name)                                  #Read image using the PIL library
        test_image = test_image.resize((50,50), Image.ANTIALIAS)     #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                              #Convert the image to numpy array
        test_image = test_image/255                                    #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)                #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image


      model=get_model(type_)[0]

      if(type_=='mal'):
         preds, pred_val = translate_malaria(model["model"].predict_proba(data), model["type"])
         final_json.append({"empty": False, "type":model["type"], 
                            "para":preds[0], 
                            "unin":preds[1],
                            "pred_val": pred_val})
      elif(type_=='brain'):
         preds, pred_val = translate_brain(model["model"].predict_proba(data), model["type"])
         final_json.append({"empty": False, "type":model["type"], 
                            "tumor":preds[0], 
                            "normal":preds[1],
                            "pred_val": pred_val})

      elif(type_=='breast'):
         print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
         preds, pred_val = translate_cancer(model["model"].predict_proba(data), model["type"])
         final_json.append({"empty": False, 
                            "type":model["type"], 
                            "can":preds[0], 
                            "norm":preds[1],
                            "pred_val": pred_val})
      elif(type_=='pnm'):
         preds, pred_val = translate_pnm(model["model"].predict_proba(data), model["type"])
         final_json.append({"empty": False, "type":model["type"], 
                            "bac":preds[0], 
                            "normal":preds[1],
                            "viral":preds[2],
                            "pred_val": pred_val})
      elif(type_=='oct'):
         preds, pred_val = translate_oct(model["model"].predict(data), model["type"])
         final_json.append({"empty": False, "type":model["type"], 
                            "cnv":preds[0], 
                            "dme":preds[1],
                            "drusen":preds[2],
                            "normal":preds[3],
                            "pred_val": pred_val})
      elif(type_=='dia_ret'):
         preds, pred_val = translate_retinopathy(model["model"].predict_proba(data), model["type"])
         final_json.append({"empty": False, "type":model["type"], 
                            "mild":preds[0], 
                            "mod":preds[1],
                            "norm":preds[2],
                            "severe":preds[3],
                            "pred_val": pred_val})
    else:
      warn = "Feeding blank image won't work. Please enter an input image to continue."
      pred_val =" "
      final_json.append({"pred_val": warn,
                         "para": " ",
                         "unin": " ",
                         "tumor": " ",
                         "normal": " ",
                         "bac": " ",
                         "viral": " ",
                         "cnv": " ",
                         "dme": " ",
                         "drusen": " ",
                         "mild": " ",
                         "mod": " ",
                         "severe": " ",
                         "norm": " "})

    K.clear_session()
    return jsonify(final_json)
  return jsonify({"empty":True})

def load_model_(model_name):
	model_name = os.path.join("weights",model_name)
	model = load_model(model_name)
	print("Model {} successfuly loaded".format(model_name))
	return model

def get_model(name = None):
  model_name = []
  if(name=='mal'):
    model_name.append({"model": load_model_("malaria.h5"), "type": name})
  elif(name=='brain'):
    model_name.append({"model": load_model_("brain_tumor.h5"), "type": name})
  elif(name=='pnm'):
    model_name.append({"model": load_model_("pneumonia.h5"), "type": name})
  elif(name=='oct'):
    model_name.append({"model": load_model_("retina_OCT.h5"), "type": name})
  elif(name=='dia_ret'):
    model_name.append({"model": load_model_("diabetes_retinopathy.h5"), "type": name})
  elif(name=='breast'):
    model_name.append({"model": load_model_("breastcancer.h5"), "type": name})
  return model_name

def translate_malaria(preds, type_):
  print("Prediction Value")
  print(preds)

  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  para_prob="Probability of the cell image to be Parasitized: {:.2f}%".format(y_proba_Class1)
  unifected_prob="Probability of the cell image to be Uninfected: {:.2f}%".format(y_proba_Class0)

  total = para_prob + " " + unifected_prob
  total = [para_prob,unifected_prob]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The cell image shows strong evidence of Malaria."
      return total,prediction
  else:
      prediction="Inference: The cell image shows no evidence Malaria."
      return total,prediction

def translate_cancer(preds, type_):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  can="Probability of the histopathology image to have cancer: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the histopathology image to be normal: {:.2f}%".format(y_proba_Class0)

  total = [can,norm]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The histopathology image shows strong evidence of Invasive Ductal Carcinoma."
      return total,prediction
  else:
      prediction="Inference: The cell image shows no evidence of Invasive Ductal Carcinoma."
      return total,prediction

def translate_brain(preds, type_):
  print("Prediction Value Brain")
  print(preds)

  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  tumor="Probability of the MRI scan to have a brain tumor: {:.2f}%".format(y_proba_Class1)
  normal="Probability of the MRI scan to not have a brain tumor: {:.2f}%".format(y_proba_Class0)

  #total = para_prob + " " + unifected_prob
  total = [tumor, normal]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The MRI Scan has brain tumor."
      return total,prediction
  else:
      prediction="Inference: The MRI Scan does not have brain tumor."
      return total,prediction

def translate_oct(preds, type_):
  print("Prediction Value OCT")
  print(preds)

  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100
  y_proba_Class3 = preds.flatten().tolist()[3] * 100

  cnv="Probability of the input image to have Choroidal Neo Vascularization: {:.2f}%".format(y_proba_Class0)
  dme="Probability of the input image to have Diabetic Macular Edema: {:.2f}%".format(y_proba_Class1)
  drusen="Probability of the input image to have Hard Drusen: {:.2f}%".format(y_proba_Class2)
  normal="Probability of the input image to be absolutely normal: {:.2f}%".format(y_proba_Class3)

  total = [cnv,dme,drusen,normal]
  
  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
  statements = ["The image has high evidence of Choroidal Neo Vascularization in the retinal pigment epithelium.",
               "The image has high evidence of Diabetic Macular Edema in the retinal pigment epithelium.",
               "The image has high evidence of Hard Drusen in the retinal pigment epithelium.",
               "The retinal pigment epithelium layer looks absolutely normal."]
  
  
  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction

def translate_pnm(preds, type_):
  print("Prediction Value PNM")
  print(preds)

  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100

  bac="Probability of the image to be Bacterial Pneumonia: {:.2f}%".format(y_proba_Class0)
  norm="Probability of the image to be Normal: {:.2f}%".format(y_proba_Class1)
  viral="Probability of the image to be Viral Pneumonia: {:.2f}%\n".format(y_proba_Class2)

  total = [bac,norm,viral]

  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2]
  statements = ["The chest X-Ray has high evidence for the presence of Bacterial Pneumonia.",
                "The chest X-Ray image is normal.",
                "The chest X-Ray has high evidence for the presence of Viral Pneumonia."]
  
  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction

def translate_retinopathy(preds, type_):
  print("Prediction Value Retinopathy")
  print(preds)

  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100
  y_proba_Class3 = preds.flatten().tolist()[3] * 100

  mild="Probability of the input image to have Mild Diabetic Retinopathy: {:.2f}%".format(y_proba_Class0)
  mod="Probability of the input image to have Moderate Diabetic Retinopathy: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the input image to be Normal: {:.2f}%".format(y_proba_Class2)
  severe="Probability of the input image to have Severe Diabetic Retinopathy: {:.2f}%".format(y_proba_Class3)

  total = [mild,mod,norm,severe]
  
  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
  statements = ["The image has high evidence for Mild Nonproliferative Diabetic Retinopathy Disease.",
               "The image has high evidence for Moderate Nonproliferative Diabetic Retinopathy Disease.",
               "The image has no evidence for Nonproliferative Diabetic Retinopathy Disease.",
               "The image has high evidence for Severe Nonproliferative Diabetic Retinopathy Disease."]
  
  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction


#Predict image using VGG16 pretrained models


app.run("0.0.0.0",80, debug = False)

