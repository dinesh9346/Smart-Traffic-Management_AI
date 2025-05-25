import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_out_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
else:
    output_layers = [layer_names[unconnected_out_layers - 1]]

# Load classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define vehicle classes
vehicle_classes = ["car", "motorbike", "bus", "truck"]

# Function to detect vehicles
def detect_vehicles(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                # Object detected is a vehicle
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Draw black box around detected vehicle
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)  # Display vehicle name above the box

    return len(indexes), class_ids

# Function to calculate average speed based on vehicle counts
def calculate_average_speed(vehicle_counts):
    return [30 for _ in vehicle_counts]

# Function to calculate dynamic signal time based on vehicle count and average speed
def calculate_signal_time(vehicle_count, average_speed):
    # Dummy implementation: signal time is proportional to vehicle count and inversely proportional to average speed
    return vehicle_count * 10 / average_speed


# Function to plot vehicle classification pie chart
def plot_vehicle_classification(class_ids, vehicle_count, green_time, orange_time, red_time, average_speed, video_index):
    class_count = defaultdict(int)
    for class_id in class_ids:
        class_count[classes[class_id]] += 1

    labels = list(class_count.keys())
    sizes = list(class_count.values())

    plt.figure(figsize=(10, 10))  # Increase the size of the pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 26, 'fontweight': 'bold'})
    plt.axis('equal')
    plt.title(f"Vehicle Classification (Video {video_index + 1})", fontsize=26, fontweight='bold')
    plt.figtext(0.5, 0.01, f"Vehicle Count: {vehicle_count}\nGreen Time: {green_time:.2f}s, Orange Time: {orange_time:.2f}s, Red Time: {red_time:.2f}s\nAverage Speed: {average_speed} km/h", ha="center", fontsize=22, fontweight='bold')
    plt.savefig(f"classification_pie_chart_{video_index}.png")
    plt.close()

# Open 2 video files
video_paths = ["video1.mp4", "video2.mp4"]
caps = [cv2.VideoCapture(path) for path in video_paths]

# Check if videos opened successfully
if not all(cap.isOpened() for cap in caps):
    print("Error: Could not open all video files.")
    exit()

# Get the dimensions of the first frame
ret, frame = caps[0].read()
if not ret:
    print("Error: Could not read frame from video.")
    exit()
height, width, _ = frame.shape

while True:
    # Read frames from each video capture
    frames = [cap.read()[1] for cap in caps]

    # Check if any frame is None (end of video)
    if any(frame is None for frame in frames):
        break

    # Resize frames to the same dimensions
    frames = [cv2.resize(frame, (width, height)) for frame in frames]

    # Detect vehicles in each frame
    vehicle_counts = []
    all_class_ids = []
    for frame in frames:
        count, class_ids = detect_vehicles(frame)
        vehicle_counts.append(count)
        all_class_ids.extend(class_ids)

    # Calculate average speed
    average_speeds = calculate_average_speed(vehicle_counts)

    # Calculate dynamic signal time
    green_times = [calculate_signal_time(vehicle_counts[i], average_speeds[i]) for i in range(len(vehicle_counts))]
    orange_times = [green_time * 0.2 for green_time in green_times]  # Example: orange time is 20% of green time
    red_times = [green_time * 0.8 for green_time in green_times]  # Example: red time is 80% of green time

    # Plot vehicle classification pie chart for each video
    for i in range(len(frames)):
        plot_vehicle_classification(all_class_ids, vehicle_counts[i], green_times[i], orange_times[i], red_times[i], average_speeds[i], i)

    # Combine frames (e.g., side by side)
    combined_frame = np.hstack(frames)

    # Resize pie charts to fit in the frame
    pie_chart_0 = cv2.imread("classification_pie_chart_0.png")
    pie_chart_1 = cv2.imread("classification_pie_chart_1.png")
    pie_chart_resized_0 = cv2.resize(pie_chart_0, (500, 500))  # Increase the size of the pie chart
    pie_chart_resized_1 = cv2.resize(pie_chart_1, (500, 500))  # Increase the size of the pie chart
    combined_frame[0:500, 0:500] = pie_chart_resized_0  # Place pie chart in the first video frame
    combined_frame[0:500, width:width+500] = pie_chart_resized_1  # Place pie chart in the second video frame

    # Show the combined frame
    cv2.imshow("Traffic Management", combined_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and close windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()