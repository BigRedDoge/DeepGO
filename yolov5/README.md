# Yolov5 Modified for Stage1_Conv output

The BaseModel class in yolo.py has been edited to output a specified layer (in this case Stage1_Conv of the yolov5s model)

To change the layer outputed, change the output_layer variable in the forward_once function to the desired layer number from the model yaml.

TODO: Make changing the desired output layer to be defined when intializing the model
