from scripts.inference import inference_once
# model_path = "mtgv/MobileVLM-1.7B" # MobileVLM
model_path = "mtgv/MobileVLM_V2-3B" # MobileVLM V2
image_file = "assets/samples/IntoFire.png"
prompt_str = "Is there fire?"
# (or) What is the title of this book?
# (or) Is this book related to Education & Teaching?

args = type('Args', (), {
    "model_path": model_path,
    "image_file": image_file,
    "prompt": prompt_str,
    "conv_mode": "v1",
    "temperature": 0, 
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "load_8bit": False,
    "load_4bit": False,
})()

inference_once(args)