#📌 Seam Carving Image Resizing
#Content-Aware Image Retargeting using Dynamic Programming and Greedy Algorithms

#📖 Overview
This project implements Seam Carving, a content-aware image resizing technique introduced by Avidan & Shamir (2007).
Instead of scaling or cropping, seam carving removes the least important pixels using:
✔ Dynamic Programming (DP) — globally optimal seams
✔ Greedy Algorithm — locally optimal seams
The project support:
Vertical + horizontal seam removal
Full image resizing
Seam visualization
Real-time animation
Energy map visualization
DP vs Greedy comparison tools
This implementation is modular, interactive, and optimized with vectorized DP for high performance.

#🚀 Features
🔹 Seam carving algorithms
- Dynamic Programming (optimal)
- Greedy (fast but suboptimal)

🔹 Visualization tools
- Draw next seam (DP & Greedy)
- Energy map & heatmap view
- Full resizing animation
- Live seam removal (step-by-step)

🔹 Interactivity
- Real-time progress bar + timer + ETA
- Seam-by-seam shrinking animation
- User choice of resize percentage

🔹 Modular code structure
- Each component is isolated for clarity:
- Energy computation
- Seam search
- Seam removal
- DP/Greedy resizing

#⚠️ NOTE : 
- Before running the program, ensure that you have installed Python and OpenCV. (pip install numpy opencv-python)
- Ensure that you have included an image into the images folder and rename it to test.jpg (there is a sample castle image)

#⭐HOW TO USE
1. Open Command Prompt(cmd) in the project directory. (Alternatively, you can just type "cmd" on the page of the Windows Explorer)
2. Once cmd is opened, type : python main.py and click enter.
3. Done! You are now presented with a main menu with resizing options.

#⭐WHAT DO THE OPTIONS DO?
1. Resize using DP
   - Asks for width & height percentage
   - Uses Dynamic Programming seam carving
   - Shows full animation of seam removal
   - Includes progress bar, elapsed time, and ETA
   - Produces the highest-quality resized image
  
2. Resize using Greedy
   - Same input as DP
   - Uses a locally optimal Greedy algorithm
   - Faster but more distortion
   - Fully animated resizing
  
3. Visualize next DP seam
   - Overlays the DP-selected seam in red
   - Shows how the optimal path avoids important regions
   - Useful for learning and debugging
  
4. Visualize next Greedy seam
   - Overlays the Greedy-selected seam
   - Typically more jagged and less optimal
   - Helps visually explain algorithmic differences
  
5. Show Energy Map
   - Displays the Sobel-based energy map
   - High energy → edges / important content
   - Low energy → background / removable areas

#🙏SPECIAL THANKS
- Thank you to the fellow contributors and project mates for coming together to make this project work!
  - Muhammad Fakhrurazi
  - Jay Lim
  - Dylan Lau
  - Edwin Leow
  - Jasper Ho

#📝 License
- Image Credits : Broadway Tower by Newton2 on Wikipedia
- Licensing & Assignment Credits : Developed as part of Singapore Institute of Technology (SIT) Coursework. All Rights Reserved. Do not copy or distribute without permission.
