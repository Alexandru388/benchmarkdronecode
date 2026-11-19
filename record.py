#!/usr/bin/env python3
import sys, time, cv2
from picamera2 import Picamera2

if len(sys.argv) < 2:
    sys.exit("Folosire: python3 record.py NUME_CLIP [durata_s] [fps]")

name = sys.argv[1]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 25
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},  # in picamera2 da ordine BGR, buna pt OpenCV
    controls={"FrameRate": fps},
))
cam.start()
time.sleep(2)  # expunere stabila

out = cv2.VideoWriter(f"{name}.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
if not out.isOpened():
    sys.exit("Nu pot crea fisierul video.")

# --- PREGĂTIM FEREASTRA DINAINTE ---
window_name = "Previzualizare Live - Apasa 'q' pentru oprire"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.moveWindow(window_name, 320, 180)  # O mutăm în centru de pe acum

# Capturăm un cadru gol/inițial doar ca să deschidem fereastra vizual pe ecran
initial_frame = cam.capture_array("main")
cv2.imshow(window_name, initial_frame)
cv2.waitKey(100) # Să lăsăm sistemul de operare să randeze fereastra poziționată

for i in (3, 2, 1):
    print(f"Start in {i}...", flush=True)
    # Putem afișa cadre și în timpul numărătorii inverse ca să vezi live în timp ce numeri!
    frame = cam.capture_array("main")
    cv2.imshow(window_name, frame)
    cv2.waitKey(1000)

print("INREGISTREZ", flush=True)

n, t0 = 0, time.time()
while time.time() - t0 < dur:
    frame = cam.capture_array("main")
    out.write(frame)
    
    # Afișează cadrul live
    cv2.imshow(window_name, frame)
    
    # Oprire opțională apăsând tasta 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    n += 1

real = time.time() - t0
out.release()
cam.stop()
cv2.destroyAllWindows()  # Închide fereastra de previzualizare

print(f"Gata: {n} cadre in {real:.1f}s = {n/real:.1f} FPS reali -> {name}.avi")
