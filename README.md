Imagenette Image Classifier using EfficientNet-B0

This project is a Streamlit-based web application that classifies images from the Imagenette dataset using a fine-tuned EfficientNet-B0 model. Simply upload an image, and the app will predict its class along with a confidence score chart!
🚀 Features
    Upload images (.jpg, .png, .jpeg) directly from your device.
    View a preview of your uploaded image.
    Predict the image class using a pre-trained EfficientNet-B0 model.
    Visualize the confidence scores for each class with a bar chart.
    Lightweight and optimized for both CPU and GPU devices.
    
🧠 Model Details
    Model Architecture: EfficientNet-B0
    Training Dataset: Imagenette (subset of ImageNet)
    Classes:
        Tench
        English Springer
        Cassette Player
        Chain Saw
        Church
        French Horn
        Garbage Truck
        Gas Pump
        Golf Ball
        Parachute
        
    Model weights must be available as best_model.pth in the project directory.
    Model loading uses timm library: timm 
