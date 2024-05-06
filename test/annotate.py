from ultralytics.data.annotator import auto_annotate

auto_annotate(data="dogs.jpg", det_model="yolov8x.pt", sam_model='sam_b.pt', device='mps')
