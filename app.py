import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

from pymongo import MongoClient

# connect the database
client = MongoClient('mongodb://localhost:27017/')
db = client['plant_database']
plants_collection = db['plants']


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load custom model with modified final layer
model = models.resnet50()
num_classes = 40  # Define the number of classes in your model
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('plant_image_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels
class_names = ['Aloe Vera','Amla','Amrutha Balli','Arali','Ashoka','Ashwagandha','Avacado','Bamboo','Basale','Betel',
               'Betel Nut','Brahmi','Castor','Curry Leaf','Doddapatre','Ekka','Ganike','Guave','Geranium','Henna',
               'Hibiscus','Honge','Insulin','Jasmine','Lemon','Lemongrass','Mango','Mint','Neem','Nithyapushpa',
               'Nooni','Papaya','Pepper','Pomegranate','Raktachandini','Rose','Sapota','Tulasi','Wood sorel']
               

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    if confidence.item() < 0.5 or predicted_class.item() >= len(class_names) or class_names[predicted_class.item()]=='Sapota':
        return None  # Return None if the image is not recognized
    else:
        return class_names[predicted_class.item()]

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for plant mode (index page)
@app.route('/plant_mode')
def plant_mode():
    return render_template('index.html')

# Route for the problem mode 
@app.route('/problemmode', methods=['GET', 'POST'])
def problemmode():
    query = None
    results = []
    if request.method == 'POST':
        query = request.form['problem']
        # Search for plants that solve the given problem
        plants = plants_collection.find({"uses.use": {"$regex": query, "$options": "i"}})
        # Filter only the relevant problems_solved entries for each plant
        for plant in plants:
            matching_uses = [
                use for use in plant["uses"]
                if query.lower() in use["use"].lower()
            ]
            if matching_uses:
                results.append({
                    "plant_name": plant["plant_name"],
                    "matching_uses": matching_uses
                })

    return render_template('problem.html', results=results, query=query)

# Route to handle image upload and classification
@app.route('/upload', methods=['POST'])
def uploadfile():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('plant_mode'))

    file = request.files['file']
    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        image_path_for_html = image_path.replace("\\", "/")

        prediction = predict_image(image_path)
        #Fetch plant details from MongoDB based on the prediction
        plant = plants_collection.find_one({"plant_name": prediction})
        if prediction:
            # If prediction is successful, render results with plant info
            return render_template(
                'results.html',
                image_path=image_path_for_html,
                prediction=prediction,
                plant=plant
            )
        else:
            # If prediction fails, redirect to error page
            return redirect(url_for('error'))

# Route for error page
@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
