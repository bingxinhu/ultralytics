from ultralytics import SAM

# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

# Run inference with bboxes prompt
model('../ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

# Run inference with points prompt
model('../ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
