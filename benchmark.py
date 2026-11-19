#!/usr/bin/env python3
import argparse, csv, glob, os, subprocess, time, gc
import cv2, numpy as np, psutil


def pi_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return float("nan")


def pi_throttled():
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
        return out.strip().split("=")[1]
    except Exception:
        return "?"


def make_detector(method, threads):
    if method == "aruco":
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

        def run(gray):
            _, ids, _ = det.detectMarkers(gray)
            return [] if ids is None else ids.flatten().tolist()
        return run

    from pupil_apriltags import Detector
    det = Detector(families="tag36h11", nthreads=threads)

    def run(gray):
        return [r.tag_id for r in det.detect(gray)]
    return run


def load_gray(path, step):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--method", choices=["aruco", "apriltag"], required=True)
    ap.add_argument("--cores", default="0,1,2,3", help="ex: 0 sau 0,1,2,3")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--step", type=int, default=1, help="ia 1 din N cadre (scade RAM si timp)")
    ap.add_argument("--target-id", type=int, default=0)
    ap.add_argument("--out", default="rezultate.csv")
    a = ap.parse_args()

    cores = [int(c) for c in a.cores.split(",")]
    os.sched_setaffinity(0, cores)          # fixeaza procesul pe nucleele alese
    cv2.setNumThreads(len(cores))
    run = make_detector(a.method, len(cores))
    proc = psutil.Process()

    new_file = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(["method", "clip", "cores", "repeat", "frames", "mean_ms", "median_ms",
                    "p95_ms", "max_ms", "fps", "det_rate", "other_ids", "cpu_pct",
                    "rss_mb", "temp_start", "temp_end", "throttled"])

    clips = sorted(glob.glob(os.path.join(a.folder, "*.avi")))
    if not clips:
        raise SystemExit(f"Nu am gasit clipuri .avi in {a.folder}")

    for path in clips:
        clip = os.path.splitext(os.path.basename(path))[0]
        frames = load_gray(path, a.step)
        if not frames:
            print(f"[sar peste] {clip}: nu pot citi cadre")
            continue

        for fr in frames[:20]:              # incalzire, nu se masoara
            run(fr)

        for rep in range(1, a.repeats + 1):
            times, hits, others = [], 0, 0
            t_start = pi_temp()
            proc.cpu_percent(None)          # porneste masurarea CPU
            for fr in frames:
                t0 = time.perf_counter()
                ids = run(fr)
                times.append((time.perf_counter() - t0) * 1000)
                if a.target_id in ids:
                    hits += 1
                others += sum(1 for i in ids if i != a.target_id)
            cpu = proc.cpu_percent(None)    # % CPU de la apelul anterior (max 100 x nr nuclee)
            t = np.array(times)
            row = [a.method, clip, len(cores), rep, len(frames),
                   round(t.mean(), 2), round(float(np.median(t)), 2),
                   round(float(np.percentile(t, 95)), 2), round(t.max(), 2),
                   round(1000 / t.mean(), 2), round(hits / len(frames), 4), others,
                   round(cpu, 1), round(proc.memory_info().rss / 1e6, 1),
                   t_start, pi_temp(), pi_throttled()]
            w.writerow(row)
            fh.flush()
            print(f"{a.method} | {clip} | {len(cores)} nuclee | rep {rep}: "
                  f"{row[5]} ms/cadru, {row[9]} FPS, detectie {row[10]*100:.0f}%")

        del frames
        gc.collect()

    fh.close()
    print(f"Gata. Rezultate in {a.out}")


if __name__ == "__main__":
    main()
