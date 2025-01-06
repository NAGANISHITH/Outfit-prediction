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
def get_occasional_outfit_suggestions(color_category, gender, occasion,season):
    # Full dataset for outfit suggestions based on color category, gender, and occasion
    outfits = {
    'Light, Pale White': {
            'male': {
                'spring': {
                    'party': 'Light blue shirt, white chinos, loafers, light scarf.',
                    'function': 'Pastel blazer, white shirt, beige trousers, brown shoes.',
                    'college': 'White t-shirt, khaki shorts, sneakers.',
                    'marriage': 'Beige suit, pastel shirt, brown belt, and shoes.',
                    'office': 'Light grey suit, white shirt, and brown shoes.'
                },
                'summer': {
                    'party': 'White linen shirt, light khaki pants, sandals.',
                    'function': 'Light blue polo, white shorts, loafers.',
                    'college': 'Casual white t-shirt, denim shorts, sneakers.',
                    'marriage': 'Light linen suit, pastel tie, brown shoes.',
                    'office': 'Light beige shirt, grey trousers, and loafers.'
                },
                'autumn': {
                    'party': 'Brown leather jacket, white shirt, dark jeans, boots.',
                    'function': 'Beige sweater, navy chinos, brown boots.',
                    'college': 'Olive hoodie, light denim jeans, sneakers.',
                    'marriage': 'Brown blazer, white shirt, khaki pants, loafers.',
                    'office': 'Grey wool suit, white shirt, black shoes.'
                },
                'winter': {
                    'party': 'Navy sweater, black jeans, and boots.',
                    'function': 'Wool coat, scarf, grey trousers, and brown shoes.',
                    'college': 'Puffer jacket, thermal jeans, boots.',
                    'marriage': 'Navy suit, white shirt, dark tie, and boots.',
                    'office': 'Charcoal suit, black shoes, and winter coat.'
                }
            },
            'female': {
                'spring': {
                    'party': 'Pastel pink dress, nude heels, and a light scarf.',
                    'function': 'Floral dress, beige sandals, pearl jewelry.',
                    'college': 'White blouse, floral skirt, and sneakers.',
                    'marriage': 'Elegant pastel gown, pearl accessories.',
                    'office': 'Pastel blouse, pencil skirt, and nude shoes.'
                },
                'summer': {
                    'party': 'Light sundress, sandals, straw hat.',
                    'function': 'Floral maxi dress, wedge sandals.',
                    'college': 'Casual top, shorts, and sneakers.',
                    'marriage': 'Linen dress, minimal accessories, flat sandals.',
                    'office': 'Sleeveless blouse, light trousers, sandals.'
                },
                'autumn': {
                    'party': 'Knit dress, brown boots, leather jacket.',
                    'function': 'Layered outfit with scarf, boots, and warm cardigan.',
                    'college': 'Sweater, jeans, and boots.',
                    'marriage': 'Velvet gown, gold jewelry, and heels.',
                    'office': 'Knit blouse, wool skirt, and boots.'
                },
                'winter': {
                    'party': 'Long-sleeve dress, tights, and boots.',
                    'function': 'Wool dress, scarf, and heeled boots.',
                    'college': 'Puffer jacket, thermal leggings, and sneakers.',
                    'marriage': 'Warm gown, faux fur wrap, and boots.',
                    'office': 'Wool coat, dress, tights, and boots.'
                }
            }
        },
        'White, Fair': {
            'male': {
                'spring': {
                    'party': 'Navy shirt, white chinos, brown loafers.',
                    'function': 'Light green blazer, white shirt, beige trousers.',
                    'college': 'Blue t-shirt, khaki shorts, white sneakers.',
                    'marriage': 'Grey suit, pastel tie, brown shoes.',
                    'office': 'Beige shirt, navy trousers, black shoes.'
                },
                'summer': {
                    'party': 'Light linen shirt, khaki pants, loafers.',
                    'function': 'White polo, beige shorts, sandals.',
                    'college': 'White t-shirt, denim shorts, sneakers.',
                    'marriage': 'Beige linen suit, pastel accessories.',
                    'office': 'Light blue shirt, grey trousers, loafers.'
                },
                'autumn': {
                    'party': 'Brown sweater, white jeans, boots.',
                    'function': 'Grey blazer, dark jeans, brown boots.',
                    'college': 'Olive jacket, jeans, sneakers.',
                    'marriage': 'Charcoal suit, white shirt, brown tie.',
                    'office': 'Grey suit, navy tie, black shoes.'
                },
                'winter': {
                    'party': 'Black sweater, dark jeans, boots.',
                    'function': 'Wool coat, scarf, grey trousers.',
                    'college': 'Puffer jacket, warm jeans, boots.',
                    'marriage': 'Navy tuxedo, black tie, boots.',
                    'office': 'Charcoal suit, winter coat, gloves.'
                }
            },
            'female': {
                'spring': {
                    'party': 'Floral dress, nude heels, pearl earrings.',
                    'function': 'Pastel top, beige trousers, sandals.',
                    'college': 'Light blouse, floral skirt, sneakers.',
                    'marriage': 'Pastel gown, pearl accessories.',
                    'office': 'Pastel shirt, pencil skirt, nude shoes.'
                },
                'summer': {
                    'party': 'Sleeveless dress, wedge sandals.',
                    'function': 'Floral dress, flat sandals, sun hat.',
                    'college': 'Casual shirt, denim shorts, sneakers.',
                    'marriage': 'Light dress, pearl earrings, flat sandals.',
                    'office': 'Light blouse, beige trousers, loafers.'
                },
                'autumn': {
                    'party': 'Brown sweater dress, leather boots.',
                    'function': 'Layered look with scarf, jeans, and boots.',
                    'college': 'Knit sweater, jeans, sneakers.',
                    'marriage': 'Velvet dress, gold earrings, heels.',
                    'office': 'Knit blouse, wool skirt, tights, and boots.'
                },
                'winter': {
                    'party': 'Warm dress, tights, and boots.',
                    'function': 'Wool coat, scarf, and ankle boots.',
                    'college': 'Puffer jacket, leggings, and sneakers.',
                    'marriage': 'Warm gown, fur wrap, and boots.',
                    'office': 'Wool coat, dress, tights, and boots.'
                }
            }
        },
                'Medium White to Light Brown': {
            'male': {
                'spring': {
                    'party': 'Light green shirt, beige chinos, brown loafers.',
                    'function': 'Tan blazer, white shirt, grey trousers.',
                    'college': 'Pastel t-shirt, khaki shorts, sneakers.',
                    'marriage': 'Light brown suit, pastel tie, black shoes.',
                    'office': 'Olive shirt, navy trousers, brown belt and shoes.'
                },
                'summer': {
                    'party': 'Light cotton shirt, beige shorts, sandals.',
                    'function': 'White polo, tan trousers, loafers.',
                    'college': 'Beige t-shirt, denim shorts, sneakers.',
                    'marriage': 'Linen suit, pastel accessories.',
                    'office': 'Light olive shirt, grey trousers, loafers.'
                },
                'autumn': {
                    'party': 'Brown jacket, white shirt, dark jeans, boots.',
                    'function': 'Wool sweater, beige trousers, brown boots.',
                    'college': 'Olive hoodie, black jeans, sneakers.',
                    'marriage': 'Tan blazer, pastel shirt, khaki trousers.',
                    'office': 'Grey suit, white shirt, black shoes.'
                },
                'winter': {
                    'party': 'Charcoal sweater, black jeans, boots.',
                    'function': 'Wool coat, dark trousers, scarf.',
                    'college': 'Puffer jacket, warm jeans, boots.',
                    'marriage': 'Navy suit, white shirt, dark tie, and boots.',
                    'office': 'Charcoal suit, black shoes, and winter gloves.'
                }
            },
            'female': {
                'spring': {
                    'party': 'Mint green dress, nude heels, pearl accessories.',
                    'function': 'Floral top, beige trousers, sandals.',
                    'college': 'Light blouse, pastel skirt, sneakers.',
                    'marriage': 'Pastel gown, diamond earrings.',
                    'office': 'Olive shirt, pencil skirt, brown shoes.'
                },
                'summer': {
                    'party': 'Sleeveless sundress, sandals, sun hat.',
                    'function': 'Floral maxi dress, wedge sandals.',
                    'college': 'Beige blouse, shorts, sneakers.',
                    'marriage': 'Linen dress, pearl jewelry.',
                    'office': 'Sleeveless shirt, light trousers, sandals.'
                },
                'autumn': {
                    'party': 'Knit dress, leather boots, scarf.',
                    'function': 'Layered outfit, jeans, and ankle boots.',
                    'college': 'Warm sweater, denim jeans, sneakers.',
                    'marriage': 'Velvet gown, gold earrings.',
                    'office': 'Wool coat, knit blouse, pencil skirt, boots.'
                },
                'winter': {
                    'party': 'Long-sleeve dress, tights, and boots.',
                    'function': 'Wool sweater, scarf, and ankle boots.',
                    'college': 'Puffer jacket, leggings, and sneakers.',
                    'marriage': 'Warm gown, faux fur wrap, and boots.',
                    'office': 'Wool coat, dress, tights, and boots.'
                }
            }
        },
        'Olive, Moderate Brown': {
            'male': {
                'spring': {
                    'party': 'Light olive shirt, beige chinos, brown loafers.',
                    'function': 'Tan blazer, white shirt, grey trousers.',
                    'college': 'Pastel t-shirt, khaki pants, sneakers.',
                    'marriage': 'Light brown suit, pastel tie, and black shoes.',
                    'office': 'Light green shirt, navy trousers, brown shoes.'
                },
                'summer': {
                    'party': 'Light linen shirt, tan shorts, sandals.',
                    'function': 'White polo, beige trousers, loafers.',
                    'college': 'Beige t-shirt, denim shorts, sneakers.',
                    'marriage': 'Tan linen suit, pastel accessories.',
                    'office': 'Light grey shirt, khaki trousers, loafers.'
                },
                'autumn': {
                    'party': 'Dark green jacket, white shirt, dark jeans, boots.',
                    'function': 'Wool coat, beige trousers, brown boots.',
                    'college': 'Olive sweater, black jeans, sneakers.',
                    'marriage': 'Charcoal suit, white shirt, navy tie.',
                    'office': 'Grey suit, white shirt, brown shoes.'
                },
                'winter': {
                    'party': 'Black turtleneck, dark jeans, boots.',
                    'function': 'Overcoat, dark trousers, scarf.',
                    'college': 'Puffer jacket, warm jeans, boots.',
                    'marriage': 'Navy suit, white shirt, dark tie, and boots.',
                    'office': 'Charcoal suit, black shoes, and gloves.'
                }
            },
            'female': {
                'spring': {
                    'party': 'Light green dress, nude heels, gold jewelry.',
                    'function': 'Floral blouse, beige trousers, sandals.',
                    'college': 'Pastel top, jeans, sneakers.',
                    'marriage': 'Mint gown, diamond earrings.',
                    'office': 'Olive shirt, pencil skirt, nude shoes.'
                },
                'summer': {
                    'party': 'Sleeveless sundress, sandals, straw hat.',
                    'function': 'Light floral dress, wedge sandals.',
                    'college': 'Casual shirt, denim shorts, sneakers.',
                    'marriage': 'Pastel maxi dress, pearl accessories.',
                    'office': 'Light blouse, beige trousers, sandals.'
                },
                'autumn': {
                    'party': 'Knit dress, leather boots, and scarf.',
                    'function': 'Layered look, jeans, and ankle boots.',
                    'college': 'Olive sweater, jeans, and sneakers.',
                    'marriage': 'Velvet dress, gold earrings, and heels.',
                    'office': 'Warm coat, knit blouse, skirt, and boots.'
                },
                'winter': {
                    'party': 'Long-sleeve dress, tights, boots.',
                    'function': 'Wool sweater, scarf, and boots.',
                    'college': 'Warm puffer jacket, leggings, and boots.',
                    'marriage': 'Warm gown, faux fur wrap, and heels.',
                    'office': 'Wool coat, dress, tights, and boots.'
                }
            }
        },
        'Brown': {
            'male': {
                'spring': {
                    'party': 'Brown cotton shirt, beige chinos, brown loafers.',
                    'function': 'Tan blazer, white shirt, dark brown trousers.',
                    'college': 'Casual t-shirt, khaki shorts, sneakers.',
                    'marriage': 'Brown suit, cream tie, black shoes.',
                    'office': 'Light brown shirt, navy trousers, brown shoes.'
                },
                'summer': {
                    'party': 'Brown linen shirt, beige shorts, sandals.',
                    'function': 'White polo, tan trousers, loafers.',
                    'college': 'Light brown t-shirt, denim shorts, sneakers.',
                    'marriage': 'Linen suit, pastel accessories.',
                    'office': 'Light brown shirt, grey trousers, loafers.'
                },
                'autumn': {
                    'party': 'Dark brown jacket, white shirt, dark jeans, boots.',
                    'function': 'Wool sweater, beige trousers, brown boots.',
                    'college': 'Brown hoodie, black jeans, sneakers.',
                    'marriage': 'Tan blazer, pastel shirt, khaki trousers.',
                    'office': 'Grey suit, white shirt, brown shoes.'
                },
                'winter': {
                    'party': 'Brown sweater, black jeans, boots.',
                    'function': 'Wool coat, dark trousers, scarf.',
                    'college': 'Puffer jacket, warm jeans, boots.',
                    'marriage': 'Navy suit, white shirt, brown tie, and boots.',
                    'office': 'Brown overcoat, grey trousers, black shoes.'
                }
            },
            'female': {
                'spring': {
                    'party': 'Light brown dress, nude heels, pearl accessories.',
                    'function': 'Floral top, beige trousers, sandals.',
                    'college': 'Casual blouse, pastel skirt, sneakers.',
                    'marriage': 'Brown gown, diamond earrings.',
                    'office': 'Light brown blouse, pencil skirt, nude shoes.'
                },
                'summer': {
                    'party': 'Brown sundress, sandals, sun hat.',
                    'function': 'Floral maxi dress, wedge sandals.',
                    'college': 'Casual t-shirt, shorts, sneakers.',
                    'marriage': 'Light brown dress, gold jewelry.',
                    'office': 'Sleeveless blouse, light trousers, sandals.'
                },
                'autumn': {
                    'party': 'Brown knit dress, leather boots, scarf.',
                    'function': 'Layered outfit, jeans, and ankle boots.',
                    'college': 'Warm sweater, denim jeans, sneakers.',
                    'marriage': 'Velvet brown gown, gold earrings.',
                    'office': 'Wool coat, knit blouse, pencil skirt, boots.'
                },
                'winter': {
                    'party': 'Long-sleeve brown dress, tights, boots.',
                    'function': 'Wool sweater, scarf, and boots.',
                    'college': 'Puffer jacket, leggings, and boots.',
                    'marriage': 'Brown gown, faux fur wrap, and boots.',
                    'office': 'Brown overcoat, dress, tights, and boots.'
                }
            }
        },
        'Dark Brown': {
            'male': {
                'spring': {
                    'party': 'Dark brown shirt, beige chinos, brown loafers.',
                    'function': 'Tan blazer, white shirt, dark brown trousers.',
                    'college': 'Casual t-shirt, khaki pants, sneakers.',
                    'marriage': 'Dark brown suit, cream tie, black shoes.',
                    'office': 'Dark brown shirt, navy trousers, brown shoes.'
                },
                'summer': {
                    'party': 'Dark brown linen shirt, beige shorts, sandals.',
                    'function': 'White polo, tan trousers, loafers.',
                    'college': 'Dark brown t-shirt, denim shorts, sneakers.',
                    'marriage': 'Dark brown linen suit, pastel accessories.',
                    'office': 'Dark brown shirt, grey trousers, loafers.'
                },
                'autumn': {
                    'party': 'Brown jacket, white shirt, dark jeans, boots.',
                    'function': 'Wool sweater, beige trousers, brown boots.',
                    'college': 'Dark brown hoodie, black jeans, sneakers.',
                    'marriage': 'Charcoal blazer, pastel shirt, khaki trousers.',
                    'office': 'Grey suit, white shirt, brown shoes.'
                },
                'winter': {
                    'party': 'Dark brown sweater, black jeans, boots.',
                    'function': 'Wool coat, dark trousers, scarf.',
                    'college': 'Puffer jacket, warm jeans, boots.',
                    'marriage': 'Dark brown overcoat, white shirt, brown tie, and boots.',
                    'office': 'Dark brown overcoat, grey trousers, black shoes.'
                }
            },
            'female': {
                'spring': {
                    'party': 'Dark brown dress, nude heels, gold accessories.',
                    'function': 'Floral top, beige trousers, sandals.',
                    'college': 'Casual blouse, pastel skirt, sneakers.',
                    'marriage': 'Dark brown gown, diamond earrings.',
                    'office': 'Dark brown blouse, pencil skirt, nude shoes.'
                },
                'summer': {
                    'party': 'Dark brown sundress, sandals, straw hat.',
                    'function': 'Floral maxi dress, wedge sandals.',
                    'college': 'Casual t-shirt, shorts, sneakers.',
                    'marriage': 'Dark brown dress, gold jewelry.',
                    'office': 'Sleeveless blouse, light trousers, sandals.'
                },
                'autumn': {
                    'party': 'Dark brown knit dress, leather boots, scarf.',
                    'function': 'Layered outfit, jeans, and ankle boots.',
                    'college': 'Warm sweater, denim jeans, sneakers.',
                    'marriage': 'Velvet dark brown gown, gold earrings.',
                    'office': 'Wool coat, knit blouse, pencil skirt, boots.'
                },
                'winter': {
                    'party': 'Long-sleeve dark brown dress, tights, boots.',
                    'function': 'Wool sweater, scarf, and boots.',
                    'college': 'Puffer jacket, leggings, and boots.',
                    'marriage': 'Dark brown gown, faux fur wrap, and boots.',
                    'office': 'Dark brown overcoat, dress, tights, and boots.'
                }
            }
        },
        'Very Dark Brown to Black': {
            'male': {
                'spring': {
                    'party': 'Black shirt, beige chinos, black loafers.',
                    'function': 'Charcoal blazer, white shirt, grey trousers.',
                    'college': 'Casual black t-shirt, khaki pants, sneakers.',
                    'marriage': 'Black suit, cream tie, black shoes.',
                    'office': 'Black shirt, navy trousers, brown shoes.'
                },
                'summer': {
                    'party': 'Black linen shirt, beige shorts, sandals.',
                    'function': 'White polo, tan trousers, black loafers.',
                    'college': 'Black t-shirt, denim shorts, sneakers.',
                    'marriage': 'Black linen suit, pastel accessories.',
                    'office': 'Black shirt, grey trousers, loafers.'
                },
                'autumn': {
                    'party': 'Black jacket, white shirt, dark jeans, boots.',
                    'function': 'Wool sweater, beige trousers, black boots.',
                    'college': 'Black hoodie, black jeans, sneakers.',
                    'marriage': 'Charcoal blazer, pastel shirt, khaki trousers.',
                    'office': 'Grey suit, white shirt, black shoes.'
                },
                'winter': {
                    'party': 'Black sweater, black jeans, boots.',
                    'function': 'Wool coat, dark trousers, scarf.',
                    'college': 'Puffer jacket, warm jeans, boots.',
                    'marriage': 'Black overcoat, white shirt, black tie, and boots.',
                    'office': 'Black overcoat, grey trousers, black shoes.'
                }
            },
            'female': {
                'spring': {
                    'party': 'Black dress, nude heels, silver accessories.',
                    'function': 'Floral top, beige trousers, sandals.',
                    'college': 'Casual blouse, pastel skirt, sneakers.',
                    'marriage': 'Black gown, diamond earrings.',
                    'office': 'Black blouse, pencil skirt, nude shoes.'
                },
                'summer': {
                    'party': 'Black sundress, sandals, straw hat.',
                    'function': 'Floral maxi dress, wedge sandals.',
                    'college': 'Casual t-shirt, shorts, sneakers.',
                    'marriage': 'Black dress, pearl accessories.',
                    'office': 'Sleeveless blouse, light trousers, sandals.'
                },
                'autumn': {
                    'party': 'Black knit dress, leather boots, scarf.',
                    'function': 'Layered outfit, jeans, and ankle boots.',
                    'college': 'Warm sweater, denim jeans, sneakers.',
                    'marriage': 'Velvet black gown, gold earrings.',
                    'office': 'Wool coat, knit blouse, pencil skirt, boots.'
                },
                'winter': {
                    'party': 'Long-sleeve black dress, tights, boots.',
                    'function': 'Wool sweater, scarf, and boots.',
                    'college': 'Puffer jacket, leggings, and boots.',
                    'marriage': 'Black gown, faux fur wrap, and boots.',
                    'office': 'Black overcoat, dress, tights, and boots.'
                }
            }
        }
       
    }

    # Get outfit suggestions based on the selected color, gender, and occasion
    occasion_outfits = outfits.get(color_category, {}).get(gender, {}).get(season, {}).get(occasion, 'No suggestions available')
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
            season = request.form['season']  # Add this line to capture season input

            outfit_suggestions = get_occasional_outfit_suggestions(rf_prediction, gender, occasion, season)

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
