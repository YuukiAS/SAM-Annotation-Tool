import sys
import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QWidget,
    QFileDialog,
    QAction,
    QPushButton,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from segment_anything import sam_model_registry, SamPredictor


class NiftiAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nifti Annotation Tool")

        # Used when creating nibabel files
        self.ID = None
        self.type = None

        # Load Nifti data
        self.nifti_data = None
        self.labels = None
        self.image_for_mask = None
        self.mask = None
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_file_action = QAction("Open Nifti File", self)
        open_file_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_file_action)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout(central_widget)
        button_layout = QHBoxLayout()

        # Create figure and canvas
        self.figure, self.ax = plt.subplots()
        self.figure.set_size_inches(10, 10)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Slice slider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(
            self.update_slice
        )  # bind slider to update_slice function
        self.slice_label = QLabel("Slice")
        layout.addWidget(self.slice_label)
        layout.addWidget(self.slice_slider)
        self.current_slice = 0

        # Timeframe slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self.update_timeframe)
        self.time_label = QLabel("Timeframe")
        layout.addWidget(self.time_label)
        layout.addWidget(self.time_slider)
        self.current_timeframe = 0

        # Add buttons for label selection
        self.label_type = 1

        self.label1_button = QPushButton("Include Point")
        self.label1_button.clicked.connect(self.set_label1)
        button_layout.addWidget(self.label1_button)

        self.label2_button = QPushButton("Exclude Point")
        self.label2_button.clicked.connect(self.set_label2)
        button_layout.addWidget(self.label2_button)

        self.erase_button = QPushButton("Erase")
        self.erase_button.clicked.connect(self.set_erase)
        button_layout.addWidget(self.erase_button)

        self.label_label = QLabel("Label Type: Inclusion")
        button_layout.addWidget(self.label_label)

        layout.addLayout(button_layout)

        # Mouse click events for annotation
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Initialize the segment anything model
        self.sam_checkpoint = "model/sam_vit_b_01ec64.pth"
        self.model_type = "vit_b"

        self.device = "cpu"

        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

        # Add button to send the annotation to model
        self.send_button = QPushButton("Segment")
        self.send_button.clicked.connect(self.segment_image)
        layout.addWidget(self.send_button)

        print("Nifti Annotation Tool initialized.")

        # Add button to save mask
        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask)
        self.save_mask_button.setEnabled(False)
        layout.addWidget(self.save_mask_button)

    def display_slice(self, mask=None):
        self.ax.clear()
        if self.nifti_data.ndim == 4:
            slice_data = self.nifti_data[
                :, :, self.current_slice, self.current_timeframe
            ]
        else:
            slice_data = self.nifti_data[:, :, self.current_timeframe]
        self.ax.imshow(slice_data, cmap="gray", origin="lower")

        labeled_points = np.where(self.labels[:, :] == 1)
        self.ax.scatter(
            labeled_points[0],
            labeled_points[1],
            c="r",
            s=20,
            label="Points of Inclusion",
        )
        labeled_points2 = np.where(self.labels[:, :] == 2)
        self.ax.scatter(
            labeled_points2[0],
            labeled_points2[1],
            c="b",
            s=20,
            label="Points of Exclusion",
        )

        if mask is not None:
            mask_nan = np.where(mask == 0, np.nan, mask)
            self.ax.imshow(mask_nan, cmap="cool", alpha=0.5, origin="lower")

        self.canvas.draw()

    def open_file_dialog(self):
        # Open file dialog to select a Nifti file
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Nifti File", "", "Nifti Files (*.nii *.nii.gz)"
        )
        if file_name:
            self.load_nifti_file(file_name)
            self.display_slice()

    def load_nifti_file(self, file_name):
        # Load Nifti file using nibabel
        # define Format of nii.gz file: LVOT_1094111.nii.gz
        self.ID = file_name.split("/")[-1].split("_")[1]
        self.type = file_name.split("/")[-1].split("_")[0]
        self.image_for_mask = None
        self.mask = None
        self.save_mask_button.setEnabled(False)
        self.nifti_data = nib.load(file_name).get_fdata()
        if self.nifti_data.ndim not in [3, 4]:
            raise ValueError("Nifti file must have 3 or 4 dimensions")

        self.labels = np.zeros(
            self.nifti_data.shape[:2]
        )  # Initialize labels for annotation

        # Set slider ranges based on data
        self.time_slider.setMaximum(self.nifti_data.shape[3] - 1)
        self.time_label.setText(
            f"Timeframe: {self.current_timeframe}/{self.nifti_data.shape[3] - 1}"
        )
        if len(self.nifti_data.shape) == 4:
            self.slice_slider.setMaximum(self.nifti_data.shape[2] - 1)
            self.slice_label.setText(
                f"Slice: {self.current_slice}/{self.nifti_data.shape[2] - 1}"
            )
        else:
            self.slice_slider.setMaximum(0)
            self.slice_label.setText("Slice: 0/0")

    def update_slice(self, value):
        self.current_slice = value
        self.display_slice()

    def update_timeframe(self, value):
        self.current_timeframe = value
        self.display_slice()

    def set_label1(self):
        self.label_type = 1
        self.label_label.setText("Label Type: Inclusion")
        print("Select point of inclusion")

    def set_label2(self):
        self.label_type = 2
        self.label_label.setText("Label Type: Exclusion")
        print("Select point of exclusion")

    def set_erase(self):
        self.label_type = 0
        self.label_label.setText("Label Type: Erase")
        print("Select point to erase")

    def on_click(self, event):
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: ({x}, {y})")

        if self.label_type in [1, 2]:
            # Perform annotation if within bounds
            if 0 <= x < self.nifti_data.shape[0] and 0 <= y < self.nifti_data.shape[1]:
                self.labels[x, y] = self.label_type
                self.display_slice()
        else:
            # Erase the neighborhood of the clicked point
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.setEnabled(False)
            QApplication.processEvents()
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if (
                        0 <= x + i < self.nifti_data.shape[0]
                        and 0 <= y + j < self.nifti_data.shape[1]
                    ):
                        self.labels[x + i, y + j] = 0
                        self.display_slice()
            self.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def segment_image(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.setEnabled(False)

        try:
            # Segment the image using the SAM model
            if self.nifti_data.ndim == 4:
                image = self.nifti_data[
                    :, :, self.current_slice, self.current_timeframe
                ]
            else:
                image = self.nifti_data[:, :, self.current_timeframe]

            image_normalized = (image - image.min()) / (image.max() - image.min()) * 255
            image_normalized = image_normalized.astype(np.uint8)
            image_final = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2RGB)

            print("Loading image into model...")
            self.predictor.set_image(image_final)

            input_point = np.where(self.labels > 0)
            input_point = np.transpose(input_point)
            print(f"Input points: {input_point}")

            y_coords = input_point[:, 0]
            x_coords = input_point[:, 1]
            input_label = self.labels[y_coords, x_coords]
            # change value 2 to 0 as point of exclusion
            input_label[input_label == 2] = 0
            print(f"Input labels: {input_label}")

            QApplication.processEvents()

            print("Predicting masks...")
            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            print("Masks predicted.")

            # We only need the last mask
            mask = masks[-1]
            mask = mask.astype(np.uint8)
            mask = mask * 255


            self.image_for_mask = image
            self.mask = mask
            self.display_slice(mask)
            self.save_mask_button.setEnabled(True)

        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            QApplication.processEvents()

    def save_mask(self):
        if self.image_for_mask is not None and self.mask is not None:
            image_dir="data/image"
            label_dir="data/label"
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            filename = f"{self.type}_{self.ID}_0000.nii.gz"

            nim_image_path = os.path.join(image_dir, filename)
            nim_label_path = os.path.join(label_dir, filename)

            nim_image = nib.Nifti1Image(self.image_for_mask, np.eye(4))
            nim_label = nib.Nifti1Image(self.mask, np.eye(4))

            nib.save(nim_image, nim_image_path)
            nib.save(nim_label, nim_label_path)
            print(f"Image saved as {nim_image_path}. Mask saved as {nim_label_path}")
        else:
            raise ValueError("No image or mask to save")


    # todo: Allow at most three masks

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NiftiAnnotationTool()
    viewer.show()
    sys.exit(app.exec_())
