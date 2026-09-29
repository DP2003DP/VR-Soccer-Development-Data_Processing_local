# VR-Soccer-Development-Data_Processing


### 🧠 Data Access and Setup

The raw EEG data is stored on Box and is NOT committed to this repository due to file size limits.

**Steps to Mount Data:**

1.  **Install Box Drive:** Ensure you have Box Drive installed and logged in.
2.  **Navigate:** Locate the data folder on Box: `Box/Reseaarch_SPORTS_VR_AI/Creative Inquiry F24/EDF Data`.
3.  **Sync Locally (Required):** Right-click the folder `Men's VR Data/` in Box Drive and select "Make Available Offline" to download the necessary files.
4.  **Set Data Path:** Before running, modify the edf_file variable to point to your local Box path (e.g., `data_path = 'C:\\Users\\<username>\\Box\\Research_SPORTS_VR_AI\\Creative Inquiry F24\\EDF Data'`). 
5. **Break down into subcategories:** Structure is `Women` or `Men` with each having a `VR` and `OnField` folder where the actual edf files are 


