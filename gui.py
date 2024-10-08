import sys
import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
    QComboBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt

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
        self.image_for_mask = None
        self.current_mask_idx = 0
        self.reset_annotation()
        self.masks = [None, None, None]  # define Generated masks
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
        segment_layout = QHBoxLayout()

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

        # Add buttons for label annotation
        self.annotate_button = QPushButton("Include/Exclude Point")
        self.annotate_button.clicked.connect(self.set_annotate)
        self.annotate_button.setToolTip("Left click to include, right click to exclude")
        button_layout.addWidget(self.annotate_button)

        self.draw_box_button = QPushButton("Draw Bounding Box")
        self.draw_box_button.clicked.connect(self.set_draw_box)
        button_layout.addWidget(self.draw_box_button)

        self.erase_button = QPushButton("Erase")
        self.erase_button.clicked.connect(self.set_erase)
        button_layout.addWidget(self.erase_button)

        self.status_label = QLabel("Annotation Type: Include/Exclude")
        button_layout.addWidget(self.status_label)

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
        segment_layout.addWidget(self.send_button, 3)

        # Add a combo box to switch between masks (up to 3)
        self.mask_selection = QComboBox()
        self.mask_selection.addItems(["Mask 1", "Mask 2", "Mask 3"])
        self.mask_selection.currentIndexChanged.connect(self.switch_mask)
        segment_layout.addWidget(self.mask_selection, 1)

        layout.addLayout(segment_layout)

        # Add button to save mask
        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask)
        self.save_mask_button.setEnabled(False)
        layout.addWidget(self.save_mask_button)

        print("Nifti Annotation Tool initialized.")

    def reset_annotation(self):
        self.annotation_mode = 1
        self.label = None
        # define xyxy format
        self.box_start = None
        self.box = None

    def visualize_slice(self, new_mask=None):
        self.ax.clear()
        if self.nifti_data.ndim == 4:
            slice_data = self.nifti_data[
                :, :, self.current_slice, self.current_timeframe
            ]
        else:
            slice_data = self.nifti_data[:, :, self.current_timeframe]
        self.ax.imshow(slice_data, cmap="gray", origin="lower")

        if self.label is not None:
            labeled_points = np.where(self.label == 1)
            self.ax.scatter(
                labeled_points[0],
                labeled_points[1],
                c="r",
                s=20,
                label="Points of Inclusion",
            )
            labeled_points2 = np.where(self.label == 2)
            self.ax.scatter(
                labeled_points2[0],
                labeled_points2[1],
                c="b",
                s=20,
                label="Points of Exclusion",
            )
        if self.box is not None and (
            self.box.get_width() != 0 or self.box.get_height() != 0
        ):
            self.ax.add_patch(self.box)

        cmaps = ["cool", "spring", "summer"]  # Different color for each mask
        # Plot other existing masks
        for idx, mask in enumerate(self.masks):
            if idx == self.current_mask_idx or mask is None:
                continue
            mask_nan = np.where(mask == 0, np.nan, mask)
            mask_nan = mask_nan * 255
            self.ax.imshow(mask_nan, cmap=cmaps[idx], alpha=0.5, origin="lower")
        # Plot the generated mask
        if new_mask is not None:
            mask_nan = np.where(new_mask == 0, np.nan, new_mask)
            mask_nan = mask_nan * 255
            self.ax.imshow(
                mask_nan, cmap=cmaps[self.current_mask_idx], alpha=0.5, origin="lower"
            )
            # Save the generated mask
            # np.save(f"mask_{self.current_mask_idx}.npy", new_mask)
        self.canvas.draw()

    def open_file_dialog(self):
        # Open file dialog to select a Nifti file
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Nifti File", "", "Nifti Files (*.nii *.nii.gz)"
        )
        if file_name:
            self.load_nifti_file(file_name)
            self.visualize_slice()

    def load_nifti_file(self, file_name):
        # Load Nifti file using nibabel
        # define Format of nii.gz file: LVOT_1094111.nii.gz
        self.ID = file_name.split("/")[-1].split("_")[1].split(".")[0]
        self.type = file_name.split("/")[-1].split("_")[0]

        self.image_for_mask = None
        self.reset_annotation()
        self.masks = [None, None, None]
        self.save_mask_button.setEnabled(False)
        self.nifti_data = nib.load(file_name).get_fdata()
        if self.nifti_data.ndim not in [3, 4]:
            raise ValueError("Nifti file must have 3 or 4 dimensions")

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
        self.visualize_slice()

    def update_timeframe(self, value):
        self.current_timeframe = value
        self.visualize_slice()

    def set_annotate(self):
        self.annotation_mode = 1
        self.status_label.setText("Annotation Type: Inclusion/Exclusion")
        print("Select point to include/exclude")

    def set_draw_box(self):
        self.annotation_mode = 2
        self.status_label.setText("Annotation Type: Bounding Box")
        self.canvas.mpl_connect("button_press_event", self.on_box_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_box_drag)
        self.canvas.mpl_connect("button_release_event", self.on_box_release)

    def set_erase(self):
        self.annotation_mode = 3
        self.status_label.setText("Annotation Type: Erase")
        print("Select point to erase")

    def on_click(self, event):
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: ({x}, {y})")

        if self.label is None:
            self.label = np.zeros(self.nifti_data.shape[:2])

        if self.annotation_mode == 1:
            # Perform annotation if within bounds
            if 0 <= x < self.nifti_data.shape[0] and 0 <= y < self.nifti_data.shape[1]:
                self.label[x, y] = 1 if event.button == 1 else 2
                self.visualize_slice()
        elif self.annotation_mode == 3:
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
                        self.label[x + i, y + j] = 0
            self.visualize_slice()
            self.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def on_box_click(self, event):
        if self.annotation_mode == 2:
            self.box_start = (event.xdata, event.ydata)
            if self.box:
                self.box.remove()  #
            self.box = Rectangle(
                self.box_start,
                0,
                0,
                linewidth=2,
                edgecolor="orange",
                facecolor="none",
                zorder=10,  # prevent box from being hidden by other elements
            )
            self.ax.add_patch(self.box)
            self.canvas.draw()

    def on_box_drag(self, event):
        if self.annotation_mode == 2 and self.box_start:
            x, y = event.xdata, event.ydata
            self.box.set_width(x - self.box_start[0])
            self.box.set_height(y - self.box_start[1])
            self.canvas.draw()

    def on_box_release(self, event):
        if self.annotation_mode == 2:
            x, y = event.xdata, event.ydata
            self.box.set_width(x - self.box_start[0])
            self.box.set_height(y - self.box_start[1])
            self.canvas.draw()
            print(
                f"Bounding box height: {abs(int(self.box.get_height()))}, width: {abs(int(self.box.get_width()))}"
            )
            self.annotation_mode = 1
            self.status_label.setText("Annotation Type: Inclusion/Exclusion")

    def switch_mask(self):
        self.current_mask_idx = self.mask_selection.currentIndex()
        self.reset_annotation()
        print(f"Switching to mask {self.current_mask_idx + 1}")
        self.visualize_slice()

    def segment_image(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.setEnabled(False)

        try:
            # Preparing input_point and input_label

            if self.label is None and self.box is None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("No input points or bounding box.")
                msg.setWindowTitle("Error")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return

            input_point = None
            if self.label is not None and np.any(self.label > 0):
                input_point = np.where(self.label > 0)

            if input_point is not None:
                input_point = np.transpose(input_point)
                print(f"Input points: {input_point}")
                y_coords = input_point[:, 0]
                x_coords = input_point[:, 1]
                input_label = self.label[y_coords, x_coords]
                # change value 2 to 0 as point of exclusion
                input_label[input_label == 2] = 0
                print(f"Input labels: {input_label}")

            # Preparing input_box

            input_box = None
            if self.box:
                x0, y0 = self.box_start
                x1, y1 = (
                    self.box_start[0] + self.box.get_width(),
                    self.box_start[1] + self.box.get_height(),
                )
                input_box = np.array([int(x0), int(y0), int(x1), int(y1)])
                # Make sure points are in the correct order
                if input_box[0] > input_box[2]:
                    input_box[0], input_box[2] = input_box[2], input_box[0]
                if input_box[1] > input_box[3]:
                    input_box[1], input_box[3] = input_box[3], input_box[1]
                print(f"Input box: {input_box}")

            # Segment the image using the SAM model
            if self.nifti_data.ndim == 4:
                image = self.nifti_data[
                    :, :, self.current_slice, self.current_timeframe
                ]
            else:
                image = self.nifti_data[:, :, self.current_timeframe]

            if self.image_for_mask is None or not np.array_equal(
                image, self.image_for_mask
            ):
                image_normalized = (
                    (image - image.min()) / (image.max() - image.min()) * 255
                )
                image_normalized = image_normalized.astype(np.uint8)
                image_final = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2RGB)

                print("Loading image into model...")
                self.predictor.set_image(image_final)
            else:
                print("Using previously loaded image.")

            QApplication.processEvents()

            print("Predicting masks...")
            if input_box is not None and input_point is not None:
                masks, _, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                    box=input_box,
                )
            elif input_box is not None:
                masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[
                        None, :
                    ],  # e.g. np.array([425, 600, 700, 875]) -> array([[425, 600, 700, 875]])
                    multimask_output=False,
                )
            elif input_point is not None:
                masks, _, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
            else:
                raise ValueError("No input points or bounding box")
            print("Masks predicted.")

            # We only need the last mask
            mask = masks[-1]
            mask = mask.astype(np.uint8)

            self.image_for_mask = image
            self.masks[self.current_mask_idx] = mask
            self.visualize_slice(new_mask=mask)
            self.save_mask_button.setEnabled(True)

        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            QApplication.processEvents()

    def save_mask(self):
        if self.image_for_mask is not None:
            image_dir = "data/image"
            label_dir = "data/label"
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            filename = f"{self.type}_{self.ID}_0000.nii.gz"

            nim_image_path = os.path.join(image_dir, filename)
            nim_image = nib.Nifti1Image(self.image_for_mask, np.eye(4))
            nib.save(nim_image, nim_image_path)

            nim_label_path = os.path.join(label_dir, filename)
            combined_mask = np.zeros(self.image_for_mask.shape[:2])
            for idx, mask in enumerate(self.masks):
                if mask is None:
                    continue
                combined_mask = np.where(mask > 0, idx + 1, combined_mask)
            # rescale the mask
            nim_label = nib.Nifti1Image(combined_mask, np.eye(4))
            nib.save(nim_label, nim_label_path)

            print(f"Image saved as {nim_image_path}. Mask saved as {nim_label_path}")
        else:
            raise ValueError("No image or mask to save")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NiftiAnnotationTool()
    viewer.show()
    sys.exit(app.exec_())
