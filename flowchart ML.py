# # Let's create a basic UML diagram for the project architecture. The diagram will illustrate the flow from image input to entity extraction and cloud storage.

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# fig, ax = plt.subplots(figsize=(10, 8))

# # Adding system components as rectangles in UML diagram
# rects = {
#     "User": (0.1, 0.7),
#     "Web Interface": (0.3, 0.7),
#     "Image Preprocessing": (0.5, 0.7),
#     "ML Model\n(Entity Extraction)": (0.7, 0.7),
#     "Vultr Object Storage": (0.5, 0.5),
#     "Vultr Compute Instances": (0.7, 0.5),
#     "Extracted\nEntity Values": (0.7, 0.3),
#     "Database": (0.5, 0.3)
# }

# # Creating rectangles
# for key, pos in rects.items():
#     ax.add_patch(mpatches.Rectangle(pos, 0.15, 0.1, fill=True, edgecolor="black", facecolor="lightblue"))
#     ax.text(pos[0] + 0.075, pos[1] + 0.05, key, fontsize=10, ha='center', va='center')

# # Adding arrows to represent data flow
# arrows = [
#     [(0.25, 0.75), (0.3, 0.75)],  # User -> Web Interface
#     [(0.45, 0.75), (0.5, 0.75)],  # Web Interface -> Image Preprocessing
#     [(0.65, 0.75), (0.7, 0.75)],  # Image Preprocessing -> ML Model
#     [(0.7, 0.65), (0.7, 0.55)],   # ML Model -> Extracted Entity Values
#     [(0.7, 0.65), (0.55, 0.55)],  # ML Model -> Vultr Compute Instances
#     [(0.5, 0.65), (0.5, 0.55)],   # Image Preprocessing -> Vultr Object Storage
#     [(0.5, 0.45), (0.5, 0.35)],   # Vultr Object Storage -> Database
#     [(0.7, 0.45), (0.7, 0.35)]    # Vultr Compute Instances -> Extracted Entity Values
# ]

# # Draw arrows
# for arrow in arrows:
#     ax.annotate('', xy=arrow[1], xytext=arrow[0],
#                 arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'))

# # Hiding axis
# ax.axis('off')

# # Showing the UML diagram
# plt.show()










####################################     PLOT 2 ###########################################




#Let's create a detailed flowchart for the process. This will have smaller, more descriptive steps including data preprocessing, model training, and storage.

fig, ax = plt.subplots(figsize=(10, 12))

# Defining positions for flowchart components
flowchart_steps = {
    "Start": (0.5, 0.95),
    "Upload Image": (0.5, 0.85),
    "Image Preprocessing": (0.5, 0.75),
    "Data Augmentation\n(if required)": (0.5, 0.65),
    "Load Pretrained\nModel (optional)": (0.5, 0.55),
    "Entity Detection\nand Extraction": (0.5, 0.45),
    "Format Prediction\n(x unit)": (0.5, 0.35),
    "Save to Vultr\nObject Storage": (0.5, 0.25),
    "Store Extracted\nData in Database": (0.5, 0.15),
    "End": (0.5, 0.05)
}

# Create rectangles for each step
for key, pos in flowchart_steps.items():
    ax.add_patch(mpatches.Rectangle((pos[0] - 0.15, pos[1] - 0.05), 0.3, 0.1, fill=True, edgecolor="black", facecolor="lightgreen"))
    ax.text(pos[0], pos[1], key, fontsize=10, ha='center', va='center')

# Define arrows connecting the steps
flow_arrows = [
    [(0.5, 0.9), (0.5, 0.85)],  # Start -> Upload Image
    [(0.5, 0.8), (0.5, 0.75)],  # Upload Image -> Image Preprocessing
    [(0.5, 0.7), (0.5, 0.65)],  # Image Preprocessing -> Data Augmentation
    [(0.5, 0.6), (0.5, 0.55)],  # Data Augmentation -> Load Pretrained Model
    [(0.5, 0.5), (0.5, 0.45)],  # Load Pretrained Model -> Entity Detection
    [(0.5, 0.4), (0.5, 0.35)],  # Entity Detection -> Format Prediction
    [(0.5, 0.3), (0.5, 0.25)],  # Format Prediction -> Save to Vultr
    [(0.5, 0.2), (0.5, 0.15)],  # Save to Vultr -> Store Extracted Data
    [(0.5, 0.1), (0.5, 0.05)]   # Store Extracted Data -> End
]

# Draw arrows
for arrow in flow_arrows:
    ax.annotate('', xy=arrow[1], xytext=arrow[0],
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'))

# Hiding axis
ax.axis('off')

# Show detailed flowchart
plt.show()
