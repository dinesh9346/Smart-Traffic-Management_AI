🚦 Smart Traffic Management using YOLOv3 and YOLO7 and OpenCV
A computer vision project that uses YOLOv3 to detect vehicles in real-time video streams and dynamically adjusts traffic signal timing based on traffic density. This system aims to reduce traffic congestion and optimize vehicle flow using AI-based techniques.

Features:
- 🚗 Real-time vehicle detection using YOLOv3 (cars, trucks, buses, bikes)
- ⏱️ Intelligent traffic signal time adjustment based on vehicle count
- 📊 Dynamic pie chart visualizations of vehicle classification
- 🎥 Multi-camera (dual video input) analysis and combined display

🧠 How It Works
1. YOLOv3 model detects vehicles in video feeds.
2. Vehicle count determines traffic congestion level.
3. Signal times (green, red, yellow) are adjusted based on density.
4. Pie charts visualize traffic distribution for each direction.

Dependencies:
1.opencv-python
2.numpy
3.matplotlib
4.roboflow

📁 Project Structure
smart-traffic-management/
│
├── app.py                # Main script for video analysis and display
├── signal_time.py        # Logic to update traffic signal timings
├── download.py           # (Optional) Roboflow dataset downloader
├── yolov3.cfg            # YOLOv3 model configuration
├── yolov3.weights        # Pretrained YOLOv3 weights
├── coco.names            # Class labels (COCO dataset)
├── requirements.txt      # Python dependencies
├── video1.mp4            # First traffic video
├── video2.mp4            # Second traffic video
└── README.md             # Project documentation

🚀 Running the Project
Place your two traffic videos in the project folder and name them:
1.video1.mp4
2.video2.mp4
the two video are used for better understanding and in future work may group of camers are used to analayse traffic especially in junctions.

Run the main script:
  python app.py

  📈 Output
        1.Vehicle count displayed on screen.
        2.Pie charts showing type of vehicles (car, bus, truck, motorbike).
        3.Adaptive signal timing printed and reflected in chart overlays.

🤖 Future Enhancements
        1.Use YOLOv8 or newer models for higher accuracy.
        2.Integrate GPS and live camera feeds.
        3.Deploy on edge devices like Raspberry Pi or Jetson Nano.
        4.Real-time cloud dashboard for city traffic authorities.

🙌 Credits
      1.YOLOv3 by Joseph Redmon et al.
      2.Roboflow for easy dataset handling.
      3.OpenCV and NumPy for computer vision and processing.

⭐ Star this repository
If this project helped you or inspired you, consider giving it a ⭐ on GitHub!

