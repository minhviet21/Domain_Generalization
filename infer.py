import streamlit as st
from dataset.transform import *
import torch
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint_path = './checkpoint_95_test_on_Real World.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=device)
start_epoch = checkpoint['epoch'] + 1
st.write(f'Checkpoint from epoch {start_epoch}.')

model = checkpoint['model']
model = model.to(device)
model.eval()

# Labels
labels = [
    "Alarm Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", 
    "Calculator", "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch", 
    "Curtains", "Desk Lamp", "Drill", "Eraser", "Exit Sign", "Fan", "File Cabinet", 
    "Flipflops", "Flowers", "Folder", "Fork", "Glasses", "Hammer", "Helmet", "Kettle", 
    "Keyboard", "Knives", "Lamp Shade", "Laptop", "Marker", "Monitor", "Mop", "Mouse", 
    "Mug", "Notebook", "Oven", "Pan", "Paper Clip", "Pen", "Pencil", "Postit Notes", 
    "Printer", "Push Pin", "Radio", "Refrigerator", "Ruler", "Scissors", "Screwdriver", 
    "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "Table", "Telephone", 
    "Toothbrush", "Toys", "Trash Can", "TV", "Webcam"
]

def predict(image):
    # Preprocess image
    image = transform(image)
    image = image.to(device)
    
    # Make prediction
    with torch.no_grad():
        indices = model(image.unsqueeze(0))
        index = torch.argmax(indices, dim=1)
    
    label = labels[index.item()]
    return label

# Streamlit app
st.title("Image Classification")
st.write("Upload an image to classify.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Predict and display label
        label = predict(image)
        st.write(f'Predicted Label: {label}')
    except Exception as e:
        st.error(f"Error loading image: {e}")
