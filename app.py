from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import webcolors
import os

app = Flask(__name__)

# Function to find the closest color name
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[2]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[0]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

# Function to get the color name
def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

# CNN-based Face Detection using OpenCV DNN Module
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Function to detect faces using CNN (using OpenCV DNN module)
def detect_faces_cnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2 - x, y2 - y))
    return faces

# Refine dominant color extraction
def refine_skin_tone(dominant_color):
    # Define RGB thresholds for skin tones
    if dominant_color[0] > 200 and dominant_color[1] > 170 and dominant_color[2] > 160:
        return 'Light, Pale White'
    elif dominant_color[0] > 220 and dominant_color[1] > 180 and dominant_color[2] > 170:
        return 'White, Fair'
    elif dominant_color[0] > 180 and dominant_color[1] > 150 and dominant_color[2] > 120:
        return 'Medium White to Light Brown'
    elif dominant_color[0] > 130 and dominant_color[1] > 100 and dominant_color[2] > 70:
        return 'Olive, Moderate Brown'
    elif dominant_color[0] > 90 and dominant_color[1] > 60 and dominant_color[2] > 40:
        return 'Brown'
    elif dominant_color[0] > 60 and dominant_color[1] > 30 and dominant_color[2] > 10:
        return 'Dark Brown'
    else:
        return 'Very Dark Brown to Black'

# K-Means for Dominant Color Extraction
def extract_dominant_color(image, k=3):
    # Convert the image to RGB (if it's in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape((-1, 3))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the dominant color as the centroid of the largest cluster
    dominant_color = kmeans.cluster_centers_[kmeans.labels_].mean(axis=0).astype(int)

    # Return the dominant color in RGB format
    skin_tone = refine_skin_tone(dominant_color)  # Refine classification based on color
    return dominant_color, skin_tone

# Function to get outfit suggestions based on color category and gender
def get_occasional_outfit_suggestions(color_category, gender, occasion):
    # Full dataset for outfit suggestions based on color category, gender, and occasion
    outfits = {
    'Light, Pale White': {
        'male': {
            'party': 'Light blue shirt, dark denim jeans, black shoes, silver watch.',
            'function': 'Soft pastels (light pink, lavender), white shirt, beige chinos, brown shoes.',
            'college': 'Casual t-shirt, comfy jeans, sneakers, simple accessories.',
            'marriage': 'Elegant light suit, white shirt, silver cufflinks.',
            'office': 'Formal light blouse, beige trousers, black shoes, subtle accessories.'
        },
        'female': {
            'party': 'Soft pastels (light pink, lavender), white blouse, light gray skirt, nude shoes, pearl accessories.',
            'function': 'Soft pastels, light pink dress, nude sandals, pearl jewelry.',
            'college': 'Casual white blouse, comfy jeans, sneakers, simple accessories.',
            'marriage': 'Elegant white gown, silver or pearl jewelry.',
            'office': 'Light blouse, pencil skirt, nude heels, simple jewelry.'
        }
    },
    'White, Fair': {
        'male': {
            'party': 'Navy blue shirt, beige chinos, brown leather shoes, brown belt.',
            'function': 'Bright colors like teal, emerald green, or coral, floral shirts, white sneakers.',
            'college': 'Casual blouse, light denim jeans, sandals, minimal accessories.',
            'marriage': 'Chic formal dress, statement necklace, heels.',
            'office': 'Navy dress shirt, grey trousers, black shoes, brown belt.'
        },
        'female': {
            'party': 'Bright colors like teal, emerald green, or coral, floral dresses, white sandals, gold jewelry.',
            'function': 'Bright floral dress, white sandals, gold jewelry.',
            'college': 'Casual blouse, comfy jeans, sandals.',
            'marriage': 'Elegant dress, statement earrings, heels.',
            'office': 'Navy blue dress, black shoes, simple accessories.'
        }
    },
    'Medium White to Light Brown': {
        'male': {
            'party': 'Dark green shirt, beige trousers, brown shoes.',
            'function': 'Teal blazer, black trousers, black shoes, silver tie.',
            'college': 'Comfortable grey t-shirt, denim jeans, casual sneakers.',
            'marriage': 'Classic black tuxedo with a white shirt, black tie.',
            'office': 'Light blue dress shirt, navy blue trousers, brown belt, formal shoes.'
        },
        'female': {
            'party': 'Emerald green dress, nude heels, silver accessories.',
            'function': 'Bright teal dress, black sandals, bold jewelry.',
            'college': 'Casual green blouse, black jeans, sneakers.',
            'marriage': 'Chic green dress, diamond or silver earrings.',
            'office': 'Formal teal blouse, navy trousers, black shoes.'
        }
    },
    'Olive, Moderate Brown': {
        'male': {
            'party': 'Olive green shirt, khaki chinos, brown leather shoes.',
            'function': 'Brown blazer, white shirt, brown chinos, brown shoes.',
            'college': 'Casual olive t-shirt, denim jeans, brown boots.',
            'marriage': 'Olive green suit, beige shirt, brown tie.',
            'office': 'Olive green dress shirt, dark grey trousers, black shoes.'
        },
        'female': {
            'party': 'Olive green dress, beige sandals, gold jewelry.',
            'function': 'Olive green blouse, beige skirt, nude heels.',
            'college': 'Casual olive t-shirt, black jeans, sneakers.',
            'marriage': 'Olive dress with golden accessories.',
            'office': 'Olive green blouse, black skirt, black shoes.'
        }
    },
    'Brown': {
        'male': {
            'party': 'Brown leather jacket, black jeans, boots.',
            'function': 'Brown blazer, beige chinos, white shirt, brown shoes.',
            'college': 'Casual brown sweater, denim jeans, boots.',
            'marriage': 'Brown tuxedo, white shirt, brown leather shoes.',
            'office': 'Brown dress shirt, grey trousers, black shoes.'
        },
        'female': {
            'party': 'Brown leather jacket, black dress, black heels.',
            'function': 'Brown dress, tan heels, simple gold jewelry.',
            'college': 'Brown cardigan, casual jeans, brown boots.',
            'marriage': 'Elegant brown gown, gold accessories.',
            'office': 'Brown blouse, black skirt, black shoes.'
        }
    },
    'Dark Brown': {
        'male': {
            'party': 'Dark brown suit, white shirt, black shoes.',
            'function': 'Dark brown blazer, white shirt, beige chinos, black shoes.',
            'college': 'Dark brown sweater, denim jeans, black shoes.',
            'marriage': 'Classic dark brown tuxedo, black bow tie.',
            'office': 'Dark brown dress shirt, grey trousers, black shoes.'
        },
        'female': {
            'party': 'Dark brown dress, gold jewelry, beige heels.',
            'function': 'Dark brown blouse, beige skirt, brown heels.',
            'college': 'Dark brown sweater, denim jeans, brown boots.',
            'marriage': 'Dark brown gown, pearl or gold jewelry.',
            'office': 'Dark brown blouse, black trousers, black shoes.'
        }
    },
    'Very Dark Brown to Black': {
        'male': {
            'party': 'Black tuxedo with bowtie, black shoes.',
            'function': 'Black suit, white shirt, black tie, black shoes.',
            'college': 'Black hoodie, denim jeans, sneakers.',
            'marriage': 'Black tuxedo, white shirt, black shoes.',
            'office': 'Black dress shirt, grey trousers, black shoes.'
        },
        'female': {
            'party': 'Black dress, black heels, silver jewelry.',
            'function': 'Black gown, diamond or silver jewelry.',
            'college': 'Black hoodie, comfy jeans, sneakers.',
            'marriage': 'Black gown, diamond jewelry, black heels.',
            'office': 'Black blouse, black skirt, black shoes.'
        }
    }
}

    # Get outfit suggestions based on the selected color, gender, and occasion
    occasion_outfits = outfits.get(color_category, {}).get(gender, {}).get(occasion, 'No suggestions available')
    return occasion_outfits

@app.route('/')
def index():
    return render_template('index.html')  # This will render the HTML page

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        # Process the image here (face detection and outfit suggestions)
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Example: Call your functions for face detection and color prediction
        faces = detect_faces_cnn(image)
        for (x, y, w, h) in faces:
            face_region = image[y:y + h, x:x + w]
            dominant_color, skin_tone = extract_dominant_color(face_region, k=1)

            # Example prediction using your classifiers
            rf_prediction = skin_tone  # Use refined skin tone instead of using classifiers
            svm_prediction = skin_tone  # Same for SVM prediction

            # Generate outfit suggestion
            gender = request.form['gender']
            occasion = request.form['occasion']
            outfit_suggestions = get_occasional_outfit_suggestions(rf_prediction, gender, occasion)

            return jsonify({
                'dominant_color': dominant_color.tolist(),
                'dominant_color_preview': f'rgb({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})',  # RGB preview
                'rf_prediction': rf_prediction,
                'svm_prediction': svm_prediction,
                'outfit_suggestions': outfit_suggestions
            })
    return jsonify({"error": "Failed to process the image"})



if __name__ == '__main__':
    app.run(debug=True)
