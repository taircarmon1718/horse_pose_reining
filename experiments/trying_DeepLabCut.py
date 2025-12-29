from pathlib import Path
import deeplabcut

video_path = Path(
    r"C:\Users\carmonta\Desktop\horse_pose_reining\data\videos\gilli_comp.mp4"
)

print("Video exists:", video_path.exists())
assert video_path.exists()



superanimal_name = "superanimal_quadruped" #@param ["superanimal_topviewmouse", "superanimal_quadruped"]
model_name = "hrnet_w32" #@param ["hrnet_w32", "resnet_50"]
detector_name = "fasterrcnn_resnet50_fpn_v2" #@param ["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn"]
pcutoff = 0.15 #@param {type:"slider", min:0, max:1, step:0.05}

videotype = video_path.suffix
scale_list = []

deeplabcut.video_inference_superanimal(
    [video_path],
    superanimal_name,
    model_name=model_name,
    detector_name=detector_name,
    videotype=videotype,
    video_adapt=True,
    scale_list=scale_list,
    pcutoff=pcutoff,
)

print("DONE")
