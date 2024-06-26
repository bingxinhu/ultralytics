from ultralytics.models.sam import Predictor as SAMPredictor
import cv2

# Create SAMPredictor
overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="sam_b.pt")
predictor = SAMPredictor(overrides=overrides)

# Set image
predictor.set_image("../ultralytics/assets/zidane.jpg")  # set with image file
predictor.set_image(cv2.imread("../ultralytics/assets/zidane.jpg"))  # set with np.ndarray
results = predictor(bboxes=[439, 437, 524, 709])
results = predictor(points=[900, 370], labels=[1])

# Reset image
predictor.reset_image()
