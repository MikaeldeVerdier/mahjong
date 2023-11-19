import coremltools as ct

from model import SSD_Model

# tfmodel = SSD_Model((300, 300, 3), 40, load="/Users/mikaeldeverdier/mahjong/save_folder/model")

iou_threshold = 0.45
conf_threshold = 0.25
labels = [
    "Bamboo 1", "Bamboo 2", "Bamboo 3", "Bamboo 4", "Bamboo 5", "Bamboo 6", "Bamboo 7", "Bamboo 8", "Bamboo 9",
    "Dot 1", "Dot 2", "Dot 3", "Dot 4", "Dot 5", "Dot 6", "Dot 7", "Dot 8", "Dot 9",
    "Character 1", "Character 2", "Character 3", "Character 4", "Character 5", "Character 6", "Character 7", "Character 8", "Character 9",
    "East Wind", "South Wind", "West Wind", "North Wind",
    "Red Dragon", "Green Dragon", "White Dragon",
    "East Flower", "South Flower", "West Flower", "North Flower",
    "East Season", "South Season", "West Season", "North Season",
    "Back"
]
num_classes = len(labels)
print(num_classes)

# Load custom Tensorflow model
model = SSD_Model((288, 512, 3), num_classes)
tfmodel = model.model

num_boxes = len(model.default_boxes)

# Convert to MLModel (coremltools)
mlmodel = ct.convert(
    tfmodel,
    inputs=[ct.ImageType("input_1", shape=(1, 512, 288, 3))]
)

spec = mlmodel.get_spec()
old_box_output_name = spec.description.output[1].name
old_scores_output_name = spec.description.output[0].name
ct.utils.rename_feature(spec, old_scores_output_name, "raw_confidence")
ct.utils.rename_feature(spec, old_box_output_name, "raw_coordinates")
spec.description.output[0].type.multiArrayType.shape.extend([num_boxes, num_classes])
spec.description.output[1].type.multiArrayType.shape.extend([num_boxes, 4])
spec.description.output[0].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE
spec.description.output[1].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

print(spec.description.output)

mlmodel = ct.models.MLModel(spec)

# Build Non Maximum Suppression model
nms_spec = ct.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3

for i in range(2):
    decoder_output = spec.description.output[i].SerializeToString()

    nms_spec.description.input.add()
    nms_spec.description.input[i].ParseFromString(decoder_output)

    nms_spec.description.output.add()
    nms_spec.description.output[i].ParseFromString(decoder_output)

nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

output_sizes = [num_classes, 4]
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
    del ma_type.shape[:]

nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

nms.iouThreshold = iou_threshold
nms.confidenceThreshold = conf_threshold
nms.pickTop.perClass = False
nms.stringClassLabels.vector.extend(labels)

nms_model = ct.models.MLModel(nms_spec)

# Assembling a pipeline model from the two models
input_features = [
    ("input_1", ct.models.datatypes.Array(3, 288, 512)),
    ("iouThreshold", ct.models.datatypes.Double()),
    ("confidenceThreshold", ct.models.datatypes.Double())
]

output_features = ["confidence", "coordinates"]

pipeline = ct.models.pipeline.Pipeline(input_features, output_features)

pipeline.add_model(mlmodel)
pipeline.add_model(nms_model)

# The "image" input should really be an image, not a multi-array
pipeline.spec.description.input[0].ParseFromString(spec.description.input[0].SerializeToString())
print(pipeline.spec.description)

# Copy the declarations of the "confidence" and "coordinates" outputs
# The Pipeline makes these strings by default
pipeline.spec.description.output[0].ParseFromString(nms_spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(nms_spec.description.output[1].SerializeToString())

# Add descriptions to the inputs and outputs
pipeline.spec.description.input[1].shortDescription = "(optional) IOU threshold override"
pipeline.spec.description.input[2].shortDescription = "(optional) Confidence threshold override"
pipeline.spec.description.output[0].shortDescription = "Boxes class confidences"
pipeline.spec.description.output[1].shortDescription = "Boxes [x, y, width, height] (normalized to [0...1])"  # "relative to screen size"?

# Add metadata to the model
pipeline.spec.description.metadata.shortDescription = "Mahjong Tile Object Detector"
pipeline.spec.description.metadata.author = "Mikael de Verdier"

# Add the default threshold values and list of class labels
user_defined_metadata = {
    "iou_threshold": str(iou_threshold),
    "confidence_threshold": str(conf_threshold)
    # "classes": ", ".join(labels)
}
pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)

# Don't forget this or Core ML might attempt to run the model on an unsupported operating system version!
pipeline.spec.specificationVersion = 3

ct_model = ct.models.MLModel(pipeline.spec)

ct_model.save("save_folder/output_model.mlpackage")
