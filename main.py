import cv2

# from resize_dp import seam_carve_to_size_dp
# from resize_greedy import seam_carve_to_size_greedy
from visualisation import (
    visualize_dp_seam,
    visualize_greedy_seam,
    animate_dp_vertical,
    animate_greedy_vertical,
    animate_resize_dp,
    animate_resize_greedy,
)


def main():
    img_path = "images/test.jpg"  # change if needed
    img = cv2.imread(img_path)

    if img is None:
        print(f"ERROR: Could not load image : {img_path}. Please check that an image is loaded inside the folder -> images/ and name is the same as img_path, then run the command again.")
        return

    H, W, _ = img.shape
    print(f"Loaded image: {W} x {H}")

    while True:
        print("\n===== Seam Carving Menu =====")
        print("1. Resize using DP")
        print("2. Resize using Greedy")
        print("3. Visualize next DP seam")
        print("4. Visualize next Greedy seam")
        print("5. Animate DP vertical seam removal")
        print("6. Animate Greedy vertical seam removal")
        print("7. Exit")
        choice = input("Enter option: ")

        if choice == "1":
            pct_w = float(input("Width % (e.g. 70): ")) / 100.0
            pct_h = float(input("Height % (e.g. 50): ")) / 100.0
            new_w = max(1, int(W * pct_w))
            new_h = max(1, int(H * pct_h))
            carved = animate_resize_dp(img.copy(), new_h, new_w, delay=10)
            cv2.imshow("Original", img)
            cv2.imshow("DP Resized", carved)
            cv2.imwrite("test results/dp_result.jpg", carved)
            cv2.waitKey(0)

        elif choice == "2":
            pct_w = float(input("Width % (e.g. 70): ")) / 100.0
            pct_h = float(input("Height % (e.g. 50): ")) / 100.0
            new_w = max(1, int(W * pct_w))
            new_h = max(1, int(H * pct_h))
            carved = animate_resize_greedy(img.copy(), new_h, new_w, delay=10)
            cv2.imshow("Original", img)
            cv2.imshow("Greedy Resized", carved)
            cv2.imwrite("test results/greedy_result.jpg", carved)
            cv2.waitKey(0)

        elif choice == "3":
            vis, energy = visualize_dp_seam(img)
            cv2.imshow("DP Seam", vis)
            cv2.waitKey(0)

        elif choice == "4":
            vis, energy = visualize_greedy_seam(img)
            cv2.imshow("Greedy Seam", vis)
            cv2.waitKey(0)

        elif choice == "5":
            n = int(input("How many seams to animate (DP)? "))
            animate_dp_vertical(img.copy(), n)

        elif choice == "6":
            n = int(input("How many seams to animate (Greedy)? "))
            animate_greedy_vertical(img.copy(), n)

        elif choice == "7":
            print("Exiting.")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
