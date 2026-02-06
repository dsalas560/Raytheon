import sys
import os
import csv
import math
import matplotlib.pyplot as plt

def dist3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_from_csv.py <trajectory.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_xy = base + "_1_xy.png"
    out_z  = base + "_2_z.png"
    out_d  = base + "_3_drift.png"

    t, x, y, z, d = [], [], [], [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("x_m","") == "" or row.get("y_m","") == "" or row.get("z_m","") == "":
                continue
            try:
                tt = float(row["t_sec"])
                xx = float(row["x_m"])
                yy = float(row["y_m"])
                zz = float(row["z_m"])
            except Exception:
                continue
            t.append(tt); x.append(xx); y.append(yy); z.append(zz)
            d.append(dist3(xx, yy, zz))

    if len(t) < 2:
        print("Not enough TRACKING samples (x_m/y_m/z_m) to plot.")
        print("Tip: open the CSV and check if x_m/y_m/z_m columns are mostly blank.")
        sys.exit(1)

    # XY
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("XY Trajectory (relative to start)")
    plt.axis("equal"); plt.grid(True)
    plt.savefig(out_xy, dpi=160)
    print("Saved:", os.path.abspath(out_xy))

    # Z vs time
    plt.figure()
    plt.plot(t, z)
    plt.xlabel("Time (s)"); plt.ylabel("Z (m)")
    plt.title("Z vs Time")
    plt.grid(True)
    plt.savefig(out_z, dpi=160)
    print("Saved:", os.path.abspath(out_z))

    # Drift vs time
    plt.figure()
    plt.plot(t, d)
    plt.xlabel("Time (s)"); plt.ylabel("Distance from start (m)")
    plt.title("Distance-from-start vs Time")
    plt.grid(True)
    plt.savefig(out_d, dpi=160)
    print("Saved:", os.path.abspath(out_d))

    # Optional pop-up windows
    plt.show()

if __name__ == "__main__":
    main()
