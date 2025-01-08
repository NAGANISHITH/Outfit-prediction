# Face Detection and Outfit Suggestions

A web-based application that uses computer vision and machine learning to detect faces, determine dominant skin tones, and provide tailored outfit suggestions. The application is built with Flask for backend processing and features an interactive frontend for easy use.

---

## Features
- **Face Detection**: Uses a CNN-based OpenCV model to accurately detect faces in uploaded images.
- **Skin Tone Analysis**: Extracts dominant colors from detected face regions using K-Means clustering and refines them into predefined skin tone categories.
- **Outfit Suggestions**: Provides personalized recommendations based on skin tone, gender, occasion, and season.
- **Interactive UI**: A user-friendly HTML and JavaScript-powered interface for uploading images and viewing suggestions.

---

## Project Structure

face-outfit-suggestions/

│

├── app.py                  # Main application logic

├── templates/

│   └── index.html          # Web interface HTML

├── static/                 # (Optional) For static assets like CSS or images

├── deploy.prototxt         # Face detection model configuration

├── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained face detection model

├── requirements.txt        # List of dependencies

└── README.md               # Project documentation


---

## Getting Started

### Prerequisites
Before starting, ensure you have the following installed:
- Python 3.8 or higher
- Flask
- OpenCV
- scikit-learn
- NumPy
- webcolors

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/face-outfit-suggestions.git
   cd face-outfit-suggestions
2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Add the required model files: Download and place the following files in the project directory:
   - deploy.prototxt (CNN model configuration)
   - res10_300x300_ssd_iter_140000.caffemodel (Pre-trained face detection model)
5. Run the application:
   ```bash
   python app.py
6. Access the application: Open your browser and navigate to http://127.0.0.1:5000.

---

## How It Works
1. **Upload Image**:
   - Users upload an image via the web interface.
2. **Face Detection**:
   - The backend processes the image to detect faces using a CNN-based model.
3. **Dominant Color Extraction**:
   - The dominant color is extracted from the detected face region.
4. **Skin Tone Categorization**:
   - The extracted color is refined into one of several predefined skin tone categories.
5. **Outfit Suggestions**:
   - Based on the skin tone, gender, occasion, and season, a tailored outfit suggestion is provided.

---

## Example Outputs
- **Dominant Color Preview**: The detected dominant color displayed as a preview block.
- **Skin Tone**: Refined description of the detected skin tone (e.g., "Light, Pale White").
- **Outfit Suggestions**:
  - **Gender**: Male/Female
  - **Occasion**: Party, Function, College, Marriage, Office
  - **Season**: Summer, Winter, Spring, Autumn
<img width="960" alt="op-1" src="https://github.com/user-attachments/assets/c0330957-4d2b-4cd5-852c-54121d86af4f" />
<img width="960" alt="op-2" src="https://github.com/user-attachments/assets/7fba953e-cefc-47c9-b411-7d826954d0e3" />
<img width="958" alt="op-3" src="https://github.com/user-attachments/assets/d9c8ea22-e3de-4ac2-90be-c7643c46446e" />

---
## Live Demo
https://outfit-prediction.onrender.com

---
## Technologies Used

### Backend:
- Flask (Python)
- OpenCV (Computer Vision)
- scikit-learn (Machine Learning)

### Frontend:
- HTML, CSS, JavaScript

### Models:
- CNN-based face detection:
  - `deploy.prototxt` (model configuration)
  - `res10_300x300_ssd_iter_140000.caffemodel` (pre-trained model weights)
- K-Means clustering for dominant color extraction

---

## Future Enhancements
1. **Improved Face Detection**:
   - Replace OpenCV DNN with a more advanced deep learning model for better accuracy.
2. **Enhanced Outfit Database**:
   - Add more diverse outfit suggestions for different cultural and personal preferences.
3. **Mobile Responsiveness**:
   - Optimize the UI for mobile devices.


---

## Contributions
Contributions are welcome! Feel free to submit a pull request or open an issue.

