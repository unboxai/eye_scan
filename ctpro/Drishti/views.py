from email.mime import image
from pyexpat.errors import messages
from django.shortcuts import render, redirect
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from Drishti.image_ops import DLModel
from django.shortcuts import render
import csv 
from django.http import HttpResponse
from Drishti.models import DrishtiUser
from django.core.files.storage import FileSystemStorage
from Drishti.models import Catscreen
import datetime
import json
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth.models import User

def mainhome(request):
    print('the main home view is reached.')
    return render(request, 'mainhome.html')

def home(request):
    print('the home view is reached. The rendering should happen.')
    return render(request, 'homect.html')

def form(request):
    print('form request reached.')
    if "username" in request.session:
        return render(request, 'form.html')
    else:
        return render(request, "new_login.html" )

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def test(request):
    print('test called')
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name')
            age = data.get('age')
            userid = data.get('userid')
            
            if name and age and userid:
                response_data = {
                    'message': 'API working and app is installed fine',
                    'name': name,
                    'age': age,
                    'userid': userid
                }
                return JsonResponse(response_data, status=200)
            else:
                return JsonResponse({'error': 'Missing required fields'}, status=400)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
    
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)




def signup(request):
    if request.method == 'POST':
        # Get the form data
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']
        name = request.POST['name']
        age = request.POST['age']

        # Create a new user
        
        user = DrishtiUser(username=username, name=name, password=password, email=email, age=age)
        user.save()

        # Display a success message
        success_message = "Signup successful! Please login."

        # Redirect to the login page with the success message as a query parameter
        return render(request, 'new_login.html', {'success_message': success_message})

    return render(request, 'signup.html')


def check_username(request):
    if request.method == 'POST':
        username = request.POST['username']
        count = DrishtiUser.objects.filter(username=username).count()
        if count > 0:
            return HttpResponse("1")
        else:
            return HttpResponse("0")


def login_view(request):
    print('login called')
    success_message = request.GET.get('success_message')
    
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        #user = authenticate(request, username=username, password=password)
        user = DrishtiUser.objects.filter(username=username, password=password)
        if len(user) == 0:
            return render(request, 'new_login.html', {'error_message': 'Invalid login credentials', 'success_message': success_message})        
        request.session['username'] = username
        request.session.modified = True
        return render(request,'form.html')
        #if user is not None:
        #    login(request, user)
        #    return redirect('form')  # Replace 'form' with the URL name of your next form page
        #else:
        #    return render(request, 'new_login.html', {'error_message': 'Invalid login credentials', 'success_message': success_message})
    return render(request, 'new_login.html', {'success_message': success_message})

M1 = DLModel()

def analyze_image(request):
    print('API called')
    if request.method == 'POST':
        # Retrieve form data
        name = request.POST.get('name')
        age = request.POST.get('age')
        visual_symptom = request.POST.get('visual-symptoms')
        # Retrieve and save the uploaded file
        if 'image' in request.FILES:
            print('image was present.')
            file = request.FILES['image']
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"image_{timestamp}.jpg"
            #fs = FileSystemStorage(location=r"ctpro\static\drishti\images")
            fs = FileSystemStorage(location=r"/home/eye_scan/ctpro/staticfiles")
            filename = fs.save(filename, file)
        else:
            print('no image found in req.')
            filename = None
        """
        # Save the form data and file in the CataractScreening model
        # Fetch the latest uploaded image from the table
        #image_url = catscreen_obj.image.url
        #basepath = r"ctpro/static/drishti/images"
        basepath = r"/home/bitnami/ctpro_april24/ctpro/staticfiles/drishti/images/"
        image_url_name = filename
        final_path = basepath + image_url_name
        """
        basepath = r"/home/eye_scan/ctpro/staticfiles"
        image_url_name = filename
        final_path = os.path.join(basepath, image_url_name)
        user_obj = list(DrishtiUser.objects.filter(username=request.session["username"]))[0]
        catscreen_obj = Catscreen(user=user_obj, name=name, age=age, visual_symptom=visual_symptom, path_of_file=final_path)
        catscreen_obj.save()
        print('Image is being sent for analysis has been saved and now being sent to model for Analysis.')
        # Load the input image
        message, score = M1.call(final_path, age, visual_symptom)
        # Save the result to the CataractScreening model
        catscreen_obj.result = message
        catscreen_obj.score = str(score)
        catscreen_obj.save()
        # Pass the message to the result.html template
        context = {'message': message}
        print(f'Anlaysis done with result as {message} and score as {score}. Ready to send on front end.')
        return HttpResponse(json.dumps({"message" : message , "score" : str( score)}))
        #return redirect('success')  # Replace 'success' with the URL name of the success page
    else:
        return render(request, 'form.html')


def analyze_image_alternate(request):
    print('Alternate api called.')
    if request.method == 'POST':
        # Retrieve form data
        try:
            username = request.POST['username']
            password = request.POST['password']
            name = request.POST.get('name')
            age = request.POST.get('age')
            visual_symptom = request.POST.get('visual-symptoms')
        except:
            return HttpResponse("Error: Bad request. All parameters are mandatory: (username, password, name, age, visual-symptoms, image)", status=400)
        user = DrishtiUser.objects.filter(username=username, password=password)
        if len(user) == 0:
            return HttpResponse("Error: Authentication failed.", status=401)    
        # Retrieve and save the uploaded file
        if 'image' in request.FILES:
            file = request.FILES['image']
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"image_{timestamp}.jpg"
            #fs = FileSystemStorage(location=r"static\drishti\images")
            fs = FileSystemStorage(location=r"/home/bitnami/ctpro_april24/ctpro/staticfiles/drishti/images")
            filename = fs.save(filename, file)
        else:
            return HttpResponse("Error: Bad request. Please include an image file in the 'image' parameter.", status=400)   
        # Save the form data and file in the CataractScreening model
        user_obj = list(DrishtiUser.objects.filter(username=username))[0]
        catscreen_obj = Catscreen(user=user_obj, name=name, age=age, visual_symptom=visual_symptom, image=filename)
        catscreen_obj.save()
        # Fetch the latest uploaded image from the table
        image_url = catscreen_obj.image.url
        #basepath = r"static/drishti/images"
        basepath = r"/home/eye_scan/ctpro/staticfiles"
        # Load the input image
        try:
            message, score = M1.call(basepath + image_url, age, visual_symptom)
        except:
            return HttpResponse("Error: Bad Request. Please make sure the image is png/jpeg/jpg format and is not corrupted.", status=400)
        # Save the result to the CataractScreening model
        catscreen_obj.result = message
        catscreen_obj.score = str(score)
        catscreen_obj.save()
        # Pass the message to the result.html template
        context = {'message': message}
        return HttpResponse(json.dumps({"message" : message , "score" : str( score)}))
        #return redirect('success')  # Replace 'success' with the URL name of the success page
    else:
        return HttpResponse("GET request is not allowed on this URL.")


