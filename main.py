import cv2
import numpy as np

from dp_seam import (
    seam_carve_to_size,
    carve_vertical_n,
    carve_horizontal_n,
    vertical_seam_dp,
    horizontal_seam_dp,
    compute_energy,
    remove_vertical_seam,
)

from visualisation import (
    draw_vertical_seam,
    draw_horizontal_seam,
    draw_multiple_seams,
    animate_vertical_carving,
)


def main():
    img_path = "images/monke.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print(f"ERROR: Could not load image: {img_path}")
        return

    H, W, _ = img.shape
    print(f"Loaded image: {W} x {H}")

    while True:
        print("\n===== SEAM CARVING MENU =====")
        print("1. Resize by percentages")
        print("2. Resize by removing N seams")
        print("3. Resize to target dimensions")
        print("4. Visualize the next vertical/horizontal seam")
        print("5. Visualize ALL next N seams on image (rainbow)")
        print("6. Animate seam carving (step-by-step)")
        print("7. Exit")
        choice = input("Enter option: ")

        # ---------------------------------------------------------
        # 1. Resize by percentage
        # ---------------------------------------------------------
        if choice == "1":
            scale_w = float(input("Enter width %: ")) / 100.0
            scale_h = float(input("Enter height %: ")) / 100.0

            new_w = max(1, int(W * scale_w))
            new_h = max(1, int(H * scale_h))

            carved = seam_carve_to_size(img.copy(), new_h, new_w)
            cv2.imshow("Original", img)
            cv2.imshow("Resized", carved)
            cv2.imwrite("result.jpg", carved)
            cv2.waitKey(0)

        # ---------------------------------------------------------
        # 2. Resize by removing N seams
        # ---------------------------------------------------------
        elif choice == "2":
            remove_cols = int(input("How many vertical seams to remove? "))
            remove_rows = int(input("How many horizontal seams to remove? "))

            carved = img.copy()
            if remove_cols > 0:
                carved = carve_vertical_n(carved, remove_cols)
            if remove_rows > 0:
                carved = carve_horizontal_n(carved, remove_rows)

            cv2.imshow("Original", img)
            cv2.imshow("Carved", carved)
            cv2.imwrite("carved.jpg", carved)
            cv2.waitKey(0)

        # ---------------------------------------------------------
        # 3. Resize to target dimensions
        # ---------------------------------------------------------
        elif choice == "3":
            new_w = int(input("Target width: "))
            new_h = int(input("Target height: "))

            carved = seam_carve_to_size(img.copy(), new_h, new_w)
            cv2.imshow("Original", img)
            cv2.imshow("Carved", carved)
            cv2.imwrite("resized_target.jpg", carved)
            cv2.waitKey(0)

        # ---------------------------------------------------------
        # 4. Visualize NEXT seam (one seam)
        # ---------------------------------------------------------
        elif choice == "4":
            direction = input("Vertical seam or Horizontal? (v/h): ")

            energy = compute_energy(img)

            if direction.lower() == "v":
                seam = vertical_seam_dp(energy)
                vis = draw_vertical_seam(img, seam)
            else:
                seam = horizontal_seam_dp(energy)
                vis = draw_horizontal_seam(img, seam)

            cv2.imshow("Energy Map", (energy / energy.max() * 255).astype(np.uint8))
            cv2.imshow("Seam Visualization", vis)
            cv2.waitKey(0)

        # ---------------------------------------------------------
        # 5. Visualize ALL next N seams (rainbow overlay)
        # ---------------------------------------------------------
        elif choice == "5":
            n = int(input("How many seams to visualize? "))
            temp = img.copy()
            seams = []

            for _ in range(n):
                energy = compute_energy(temp)
                seam = vertical_seam_dp(energy)
                seams.append(seam)
                temp = remove_vertical_seam(temp, seam)

            vis = draw_multiple_seams(img, seams)
            cv2.imshow("Multiple Seams Visualization", vis)
            cv2.imwrite("all_seams.jpg", vis)
            cv2.waitKey(0)

        # ---------------------------------------------------------
        # 6. Animate seam carving (step-by-step)
        # ---------------------------------------------------------
        elif choice == "6":
            n = int(input("How many seams to animate? "))
            animate_vertical_carving(img.copy(), n, delay=40)

        # ---------------------------------------------------------
        # Exit
        # ---------------------------------------------------------
        elif choice == "7":
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
