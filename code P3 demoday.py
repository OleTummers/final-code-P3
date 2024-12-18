import os
import cv2
import numpy as np
import mediapipe as mp
import time
from diffusers import StableDiffusionPipeline
import torch
from threading import Thread #no longer needed

# Stable Diffusion setup
model_directory = r"c:\AI\stable-diffusion-webui"#path
try:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, requires_safety_checker=False, safety_checker = None).to("cuda")#otherwise it makes black screens because of NSFW (even though it doesnt make inapropiate material)
    pipe.enable_attention_slicing()
    print("Model loaded successfully from local directory.")
except Exception as e:
    print(f"Error loading model from local directory: {e}. Switching to Hugging Face repository.")
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    # pipe.enable_attention_slicing()
    print("Model loaded successfully from Hugging Face.")

# MediaPipe Hands and Pose setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7, static_image_mode=False)
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

# Initialize variables
canvas = None
prev_points = {'Left': None, 'Right': None}
paths = {'Left': [], 'Right': []}
ai_generated_image = None
last_generation_time = time.time()
current_emotion = []
prev_hand_positions = {'Left': None, 'Right': None}

# Gesture-to-prompt mapping objects
gesture_prompts = {
    'circle': ['moon', 'sun', 'football', 'bubble'],
    'line': ['horizon', 'tree', 'pathway', 'road', 'railway'],
    'triangle': ['mountain', 'pyramid'],
    'square': ['building']
}

# Emotion prompt definitions
emotion_prompts = {
    "happiness": "disco exploding colorful",
    "anger": "red powerful",
    "sadness": "depressing cold",
    "peacefulness": "pastel colors"
}


# Emotion weights
emotion_weights = {
    "happiness": 0,
    "anger": 0,
    "sadness": -1,
    "peacefulness": 0
}

# Update emotion weights based on detected emotions
def update_emotion_weights(emotions):
    for emotion in emotions:
        if emotion in emotion_weights:
            emotion_weights[emotion] += 1  # Increment the count for emotion


# functions for shape detection
def get_shape_area_regions(x_coords, y_coords):
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    regions = []
    if min_y < canvas_height / 2 and min_x < canvas_width / 2:
        regions.append('top left')
    if max_y >=  canvas_height / 2 and min_x < canvas_width / 2:
        regions.append('bottom left')
    
    if min_y < canvas_height / 2 and max_x >= canvas_width / 2:
        regions.append('top right')
    if max_y >= canvas_height / 2 and max_x >= canvas_width / 2:
        regions.append('bottom right')

    return regions

#circle
def is_circle(x, y):
    center_x, center_y = np.mean(x), np.mean(y)
    distances = [np.sqrt((xi - center_x)*2 + (yi - center_y)*2) for xi, yi in zip(x, y)]
    return max(distances) - min(distances) < 55

#line
def is_line(x, y):
    if len(x) < 2:
        return False
    coeffs = np.polyfit(x, y, 1)
    return abs(coeffs[0]) > 0.3

#triangle
def is_triangle(x, y):
    if len(x) < 3:
        return False
    points = list(zip(x, y))
    distances = [
        (np.linalg.norm(np.array(points[i]) - np.array(points[j])), i, j)
        for i in range(len(points)) for j in range(i + 1, len(points))
    ]
    distances.sort(reverse=True)
    d1, i1, j1 = distances[0]
    d2, i2, j2 = distances[1]
    d3, _, _ = distances[2]

    if not (d2 + d3 > d1 and d1 + d3 > d2 and d1 + d2 > d3):
        return False
    angle1 = np.arccos((d2*2 + d3*2 - d1*2) / (2 * d2 * d3)) if d2 * d3 != 0 else 0
    angle2 = np.arccos((d1*2 + d3*2 - d2*2) / (2 * d1 * d3)) if d1 * d3 != 0 else 0
    angle3 = np.arccos((d1*2 + d2*2 - d3*2) / (2 * d1 * d2)) if d1 * d2 != 0 else 0

    if not (0.3 < angle1 < 2.8 and 0.3 < angle2 < 2.8 and 0.3 < angle3 < 2.8):
        return False

    x_spread = max(x) - min(x)
    y_spread = max(y) - min(y)
    if x_spread < 100 or y_spread < 100:
        return False

    angles = sorted([angle1, angle2, angle3])
    if angles[0] < 0.5 or angles[2] > 2.5:
        return False

    return True

#square
def is_square(x, y):
    if len(x) < 4:
        return False
    x_diff = max(x) - min(x)
    y_diff = max(y) - min(y)
    return abs(x_diff - y_diff) < 50 and x_diff > 50 and y_diff > 50

#location of objects
def detect_gesture_with_location(path):
    if len(path) < 10:
        return []
    x_coords, y_coords = zip(*path)
    detected_shapes = []
    if is_circle(x_coords, y_coords):
        detected_shapes.append(('circle', get_shape_area_regions(x_coords, y_coords)))
    if is_line(x_coords, y_coords):
        detected_shapes.append(('line', get_shape_area_regions(x_coords, y_coords)))
    if is_triangle(x_coords, y_coords):
        detected_shapes.append(('triangle', get_shape_area_regions(x_coords, y_coords)))
    if is_square(x_coords, y_coords):
        detected_shapes.append(('square', get_shape_area_regions(x_coords, y_coords)))
    return detected_shapes

#base prompt with shape
def generate_prompt_from_shapes(shapes_with_locations):
    prompt_parts = []
    for shape, regions in shapes_with_locations:
        regions_text = " and ".join(regions)
        shape_description = f"a {np.random.choice(gesture_prompts[shape])} located at the {regions_text}"
        prompt_parts.append(shape_description)
    return " image with " + " ".join(prompt_parts)

#adding emotions to prompt
def generate_emotion_prompt(base_prompt, emotions):
    if emotions:
        # Sort emotions by their weights
        sorted_emotions = sorted(emotions, key=lambda e: emotion_weights.get(e, 0), reverse=True)

        # Calculate total weight 
        total_weight = sum(emotion_weights.get(e, 0) for e in sorted_emotions)
        if total_weight == 0:
            total_weight = 1  

        # Generate weighted phrases
        weighted_phrases = []
        for emotion in sorted_emotions:
            # Calculate the weight for each emotion
            weight = emotion_weights.get(emotion, 0)
            percentage = 1+(weight / total_weight) #so that stable diffusion gets it 
            weighted_phrases.append(f"({emotion_prompts.get(emotion, emotion)}: {percentage:.1f})")

        # Combine base prompt with weighted emotions
        return "A " + " ".join(weighted_phrases) + base_prompt

    return base_prompt

# directory where images are saved
output_directory = r"c:\Users\20234276\Desktop\Generated_Images"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#generate image
def generate_image_with_prompt(prompt):
    try:
        num_inference_steps = 20
        guidance_scale = 7.5
        resolution = 512
        generated_image = pipe(prompt, 
                               num_inference_steps=num_inference_steps, 
                               height=resolution, 
                               width=resolution,
                               guidance_scale=guidance_scale).images[0]
        
        # Save the generated image to directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_directory, f"generated_image_{timestamp}.png")
        generated_image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        return cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

#for anger
def detect_fist_clenching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
    return distance < 0.05

def detect_clumsy_fast_movements(current_pos, prev_pos):
    if prev_pos is None:
        return False
    movement_speed = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    return movement_speed > 0.15  # Fast or clumsy movement

#for peacefulness
def detect_slow_controlled_movements(current_pos, prev_pos):
    if prev_pos is None:
        return False
    movement_speed = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    return movement_speed < 0.05  # Slow or controlled movement

# for sadness
def detect_face_covering_with_hands(left_wrist, right_wrist, nose):
    return (left_wrist.y < nose.y + 0.1 and left_wrist.x < nose.x + 0.15 and left_wrist.x > nose.x - 0.15) or \
           (right_wrist.y < nose.y + 0.1 and right_wrist.x < nose.x + 0.15 and right_wrist.x > nose.x - 0.15)

def detect_one_hand_drawing(left_hand_active, right_hand_active):
    return left_hand_active != right_hand_active

def detect_low_arm_movement(left_wrist, right_wrist):
    return left_wrist.y > 0.8 or right_wrist.y > 0.8  # arms close to the ground

# separate window for emotion display
emotion_frame = np.zeros((300, 600, 3), dtype=np.uint8)  # Black canvas for emotion text display

# emotion display variables
cumulative_emotions = set()  # To store detected emotions within the 8-second window
emotion_display_list = []  # To store the list of emotions to display in the second window

# list to store detected emotions
emotion_history = []

# Max number of emotions to display at once
max_emotions_to_display = 30

# generation time variables 
generation_time = 0  

# Countdown variables
countdown_start_time = None
countdown_active = False
countdown_duration = 8  # 8 seconds

#  displaying detected shapes variables
shape_window = np.zeros((710, 600, 3), dtype=np.uint8)  # Black canvas
detected_shapes_history = []  # List to store detected shapes and their locations

# flag to track the "Loading..." state
loading_message_active = False

# separate window for displaying emotion percentages
emotion_percentage_window = np.zeros((300, 300, 3), dtype=np.uint8)  # Black canvas
emotion_colors = {
    "happiness": (0, 255, 255),
    "anger": (255, 0, 255),
    "sadness": (255, 255, 0),
    "peacefulness": (0, 255, 0),
}


# separate window for displaying the generated prompt
prompt_window = np.zeros((300, 1000, 3), dtype=np.uint8)  # Black canvas
current_prompt_text = ""  # To store the current prompt text

# separate window for displaying the text Body-to-Image
body_to_image_window = np.zeros((300, 500, 3), dtype=np.uint8)  # Black canvas
body_to_image_text = "Body-to-Image" 

# seperate window for black screen
black_window = np.zeros((100, 100, 3), dtype=np.uint8)


# variables for loading animation
loading_dots = 1  # Start with one dot
loading_cycle_time = 0.3  # Time in seconds for each dot cycle
last_loading_update = time.time()  # Track the last update time


# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    canvas_height, canvas_width, _ = frame.shape  # Update canvas dimensions

    results_hands = hands.process(rgb_frame)
    results_pose = pose.process(rgb_frame)

    current_emotion = []

    if results_pose.pose_landmarks:
        left_wrist = None
        right_wrist = None
        nose = None
        try:
            left_wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        except AttributeError:
            pass

        if left_wrist and right_wrist and nose:
            if left_wrist.y < 1.2 * nose.y and right_wrist.y < 1.2 * nose.y:
                current_emotion.append('happiness')

        if 'happiness' in current_emotion:
            # If happiness is detected, don't detect sadness (otherwise it detects hapiness and sadness at the same time???)
            pass
        else:
            if detect_face_covering_with_hands(left_wrist, right_wrist, nose):
                current_emotion.append('sadness')

            if detect_low_arm_movement(left_wrist, right_wrist):
                current_emotion.append('sadness')

    left_hand_active = False
    right_hand_active = False


    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # Get hand label: 'Left' or 'Right'
            hand_label = results_hands.multi_handedness[idx].classification[0].label
            hand_label = 'Left' if hand_label == 'Left' else 'Right'

            # Get index finger tip position
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * canvas_width)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * canvas_height)

            # Update paths
            paths[hand_label].append((x, y))

            if detect_fist_clenching(hand_landmarks):
                current_emotion.append('anger')

            hand_position = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)

            if hand_label == 'Left':
                left_hand_active = True
            else:
                right_hand_active = True

    if detect_one_hand_drawing(left_hand_active, right_hand_active):
        current_emotion.append('sadness')

    if 'anger' not in current_emotion:
        if len(current_emotion) == 0:
            current_emotion.append('peacefulness')

    #emotion weight
    update_emotion_weights(current_emotion)


    # Add detected emotions to cumulative_emotions
    cumulative_emotions.update(current_emotion)

    # Add detected emotion(s) to emotion history
    if current_emotion:
        for emotion in current_emotion:
            emotion_history.insert(0, emotion)  # Insert at the beginning

    # Keep only the most recent 10 emotions in history
    if len(emotion_history) > max_emotions_to_display:
        emotion_history = emotion_history[:max_emotions_to_display]
        

    # Start the countdown if it's time to generate an image
    if (not countdown_active) and (time.time() - last_generation_time > 5):
        countdown_start_time = time.time()
        countdown_active = True


    # Countdown logic
    if countdown_active:
        elapsed_time = time.time() - countdown_start_time
        remaining_time = max(0, countdown_duration - elapsed_time)

        # Create a black canvas for the countdown bar
        countdown_window = np.zeros((300, 1000, 3), dtype=np.uint8)

        # Calculate the bar's width based on the remaining time
        bar_width = int((countdown_duration - remaining_time) / countdown_duration * 1000)

        # Draw the bar
        cv2.rectangle(countdown_window, (0, 40), (bar_width, 60), (255, 0, 255), -1)  # Green bar
        cv2.rectangle(countdown_window, (0, 40), (1000, 60), (255, 255, 255), 2)  # White border

        # Check if it's 1.5 seconds away from finishing
        show_loading_message = remaining_time <= 1.5 and remaining_time > 0

        # Animate the "Loading..." message
        if remaining_time <= 1.5 and remaining_time > 0:  # Show "Loading..." near the end of the countdown
             if time.time() - last_loading_update >= loading_cycle_time:  # Update dots based on cycle time
                 loading_dots = (loading_dots % 3) + 1  # Cycle dots from 1 to 3
                 last_loading_update = time.time()  # Reset update time

             loading_message = "Loading" + "." * loading_dots  # Create animated dots
             cv2.putText(countdown_window, loading_message, (375, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Show countdown window
        cv2.imshow("Countdown Timer", countdown_window)

        # Check if the countdown is complete
        if remaining_time <= 0:
            countdown_active = False  # Stop the countdown
            # Proceed to image generation
            if paths['Left'] or paths['Right']:
                detected_shapes_left = detect_gesture_with_location(paths['Left'])
                detected_shapes_right = detect_gesture_with_location(paths['Right'])
                detected_shapes = detected_shapes_left + detected_shapes_right

                # Add shapes to the display history for new window
                detected_shapes_history = detected_shapes.copy()

                # Generate the prompt
                if detected_shapes:
                 base_prompt = generate_prompt_from_shapes(detected_shapes)
                 full_prompt = generate_emotion_prompt(base_prompt, list(cumulative_emotions))
                else:
                     full_prompt = generate_emotion_prompt(" abstract image", list(cumulative_emotions))

                # Update the prompt text for display window
                current_prompt_text = f"Prompt: {full_prompt}"

                # Print the generated prompt in console
                print(current_prompt_text)

                # Generate the image
                start_time = time.time()
                ai_generated_image = generate_image_with_prompt(full_prompt)
                generation_time = time.time() - start_time

                last_generation_time = time.time()

                paths['Left'].clear()
                paths['Right'].clear()

                # Clear cumulative_emotions after generating prompt
                cumulative_emotions.clear()

                emotion_history.clear()

                # Reset emotion weights
                for key in emotion_weights:
                    emotion_weights[key] = 0

    # Continuously update detected shapes while drawing
    # Detect shapes and locations while drawing 
    detected_shapes_left = detect_gesture_with_location(paths['Left'])
    detected_shapes_right = detect_gesture_with_location(paths['Right'])
    detected_shapes = detected_shapes_left + detected_shapes_right

    # Add detected shapes to the history
    detected_shapes_history = detected_shapes.copy()

    shape_window.fill(0)  # Clear the previous shapes
    y_offset = 30  # Start position for displaying shapes
    for shape, locations in detected_shapes_history:
     location_text = " and ".join(locations)
    
     # First line: Shape
     shape_text = f"Shape: A {shape.capitalize()}"
     cv2.putText(shape_window, shape_text, (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    
     # Second line: Location
     location_text_full = f"Location: The {location_text}"
     cv2.putText(shape_window, location_text_full, (10, y_offset + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    
     y_offset += 50  # Move down for the next shape 

    # Show the detected shapes window while drawing 
    cv2.imshow("Detected Shapes", shape_window)

             

    if ai_generated_image is not None:
        ai_generated_image_resized = cv2.resize(ai_generated_image, (canvas_width, canvas_height))
        frame[0:canvas_height, 0:canvas_width] = ai_generated_image_resized

    # skeleton on frame
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # skeleton with increased thickness 
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255, 20), thickness=2, circle_radius=5),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255, 20), thickness=2)
        )

    # Display emotion history in a new window
    emotion_window = np.zeros((1000, 150, 3), dtype=np.uint8)  # black canvas
    y_offset = 30  # Start position for displaying emotions

    for i, emotion in enumerate(emotion_history):
        cv2.putText(emotion_window, emotion, (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += 30  # Move down for the next emotion

    cv2.imshow("Detected Emotions", emotion_window)  # Display the emotion window

    # window to show the generation time
    timer_window = np.zeros((100, 300, 3), dtype=np.uint8)  # Black canvas 
    timer_window.fill(0)  # Clear timer window
    cv2.putText(timer_window, f"Gen Time: {generation_time:.2f} sec", (10, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Generation Timer", timer_window)  # Display timer window

    # Display main video feed
    cv2.imshow("Emotion Detection and Gesture Recognition", frame)

     # Calculate emotion percentages
    total_weight = sum(emotion_weights.values())
    if total_weight == 0:
         total_weight = 1  

    # Normalize weights to percentages
    emotion_percentages = {
      emotion: (weight / total_weight) * 100 for emotion, weight in emotion_weights.items()
      }

     # show emotion percentages
    emotion_percentage_window.fill(0)  # Clear the previous percentages
    bar_width = 50
    x_offset = 10
    y_offset = 30

    for emotion, percentage in emotion_percentages.items():
    # Draw bar for each emotion
         bar_length = int(percentage)  # Scale percentage to bar length
         color = emotion_colors.get(emotion, (255, 255, 255)) 

         # Draw the bar
         cv2.rectangle(emotion_percentage_window, (x_offset, y_offset), (x_offset + bar_length, y_offset + 20), color, -1)

         # Add emotion name and percentage text
         text = f"{emotion.capitalize()}: {percentage:.1f}%"
         cv2.putText(emotion_percentage_window, text, (x_offset + bar_length + 10, y_offset + 15),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

         y_offset += 40  # Move down for the next emotion

    # Display emotion percentage window
    cv2.imshow("Emotion Percentages", emotion_percentage_window)

    # Display generated prompt in separate window
    prompt_window.fill(0)  # Clear previous text
    lines = current_prompt_text.split(" ")  # Space bar
    line_length = 100  # Maximum characters per line
    y_offset = 20  # Starting y-offset for text display
    x_offset = 10  # Starting x-offset for text display

    # Wrap text to fit window width
    wrapped_lines = []
    while lines:
     line = []
     while lines and len(" ".join(line + [lines[0]])) <= line_length:
         line.append(lines.pop(0))
     wrapped_lines.append(" ".join(line))

    # Display each wrapped line on prompt window
    for i, line in enumerate(wrapped_lines):
     cv2.putText(prompt_window, line, (x_offset, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 1)

    # Show prompt window
    cv2.imshow("Generated Prompt", prompt_window)


    body_to_image_window = np.zeros((400, 500, 3), dtype=np.uint8)  # Black canvas
    cv2.putText(body_to_image_window, body_to_image_text, (110, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)#Body-to-Image!!!!!!

    # Display Body-to-Image window
    cv2.imshow("Body_to_Image", body_to_image_window)

    black_window = np.zeros((400, 500, 3), dtype=np.uint8)  # Black canvas
    # Display black window
    cv2.imshow("black", black_window)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()