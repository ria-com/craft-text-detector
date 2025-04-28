# import craft functions
import sys
sys.path.append("./")
from craft_text_detector import (
    read_image,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
from craft_text_detector.trt_models.craftnet import CraftTrtInference
from craft_text_detector.trt_models.refinenet import RefineNetTrtInference
from craft_text_detector.predict_trt import get_prediction

# set image path and export folder directory
image = 'figures/idcard.png' # can be filepath, PIL image or numpy array
output_dir = 'outputs_trt/'

# read image
image = read_image(image)

# load models
refine_net = RefineNetTrtInference("/mnt/raid/var/www/modelhub-client-trt/data/models/craft/refinenet_fixed.trt")
craft_net = CraftTrtInference("/mnt/raid/var/www/modelhub-client-trt/data/models/craft/craft_mlt_25k_2020-02-16.trt")

# perform prediction
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=True,
    long_size=1280
)

# export detected text regions
exported_file_paths = export_detected_regions(
    image=image,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True
)

# export heatmap, detection points, box visualization
export_extra_results(
    image=image,
    regions=prediction_result["boxes"],
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir
)

# unload models from gpu
empty_cuda_cache()
