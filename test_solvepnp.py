#!/usr/bin/env python3
import sys
import cv2
import numpy as np

# Verificare argumente
if len(sys.argv) < 4:
    print("Folosire: python3 test_solvepnp.py <fisier.avi> <metoda: aruco/apriltag> <distanta_reala_metri>")
    print("Exemplu: python3 test_solvepnp.py benchmark/static_1m.avi aruco 1.0")
    sys.exit(1)

video_path = sys.argv[1]
method = sys.argv[2].lower()
real_distance = float(sys.argv[3])

# Setări calibrare și dimensiuni marker (în metri)
CALIBRATION_FILE = "/home/alex/aruco_calibration/camera_calibration.yaml"

# Dimensiunile specificate de tine
# ArUco = 16.5 cm (0.165 m), AprilTag = 15.9 cm (0.159 m)
if method == "aruco":
    marker_size = 0.165  
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
elif method == "apriltag":
    marker_size = 0.159  
    # Notă: Dacă folosești un alt dicționar AprilTag (ex: tag36h11), asigură-te că îl schimbi aici
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
else:
    sys.exit("Metodă necunoscută. Folosește 'aruco' sau 'apriltag'.")

# Încărcare matrice de calibrare cameră
fs = cv2.FileStorage(CALIBRATION_FILE, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    sys.exit(f"Nu pot deschide fișierul de calibrare: {CALIBRATION_FILE}")
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("dist_coeffs").mat()
fs.release()

if camera_matrix is None or dist_coeffs is None:
    sys.exit("Fișierul de calibrare nu conține camera_matrix sau dist_coeffs.")

# Deschidere fișier video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"Nu pot deschide fișierul video: {video_path}")

# Coordonatele 3D ale colțurilor markerului în sistemul propriu al markerului (Centrul este 0,0,0)
# Ordine: sus-stânga, sus-dreapta, jos-dreapta, jos-stânga
half = marker_size / 2.0
obj_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

measured_distances = []
frames_processed = 0
frames_detected = 0

print(f"Procesare {video_path} folosind {method.upper()} (Dimensiune marker: {marker_size}m)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frames_processed += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(corners) > 0:
        # Luăm primul marker detectat
        marker_corners = corners[0][0]
        
        # Aplicăm solvePnP pentru a afla vectorii de rotație și translație
        success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, dist_coeffs)
        
        if success:
            # Distanța Z este a treia componentă din tvec (translația pe axa optică față de cameră)
            distance_z = tvec[2][0]
            measured_distances.append(distance_z)
            frames_detected += 1

cap.release()

# Calcul rezultate finale și erori
if frames_detected > 0:
    mean_measured = np.mean(measured_distances)
    std_measured = np.std(measured_distances)
    absolute_error = abs(mean_measured - real_distance)
    error_cm = absolute_error * 100.0

    print("\n--- REZULTATE SOLVE_PNP ---")
    print(f"Cadre totale: {frames_processed}")
    print(f"Cadre cu marker detectat: {frames_detected} ({frames_detected/frames_processed*100:.1f}%)")
    print(f"Distanța reală cunoscută: {real_distance:.2f} m")
    print(f"Distanța medie măsurată:   {mean_measured:.3f} m")
    print(f"Deviația standard (jitter):{std_measured * 100:.2f} cm")
    print(f"Eroarea absolută medie:    {error_cm:.2f} cm")
else:
    print("Nu s-a putut detecta niciun marker în acest clip cu metoda selectată.")
