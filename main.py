from fastapi import FastAPI, Request, Response, BackgroundTasks
from configparser import ConfigParser

import utils.image_processing as ip
import utils.SDApi as sda 
import utils.upscale as u

import uuid
import json
import time
import cv2
import os



app = FastAPI()
config = ConfigParser()
config.read("./configs/main.ini")
SDApi = sda.SDApi(url=config["API"]["url"])

api_tokens = [str(i) for i in config['API']['api_tokens'].split(',')]

files_dir_path = config['SERVER']['files_dir_path']
output_files_path = config['SERVER']['output_files_path']
del_delay = int(config['SERVER']['delete_delay'])

models_list = [str(i) for i in config['API']['models'].split(',')]


def remove_file_delay(file_peth:str, delay):
    time.sleep(delay)
    if os.path.isfile(file_peth):
        os.remove(file_peth)


@app.post("/txt2img")
async def txt2img(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "prompt" not in body.keys():
        return Response(json.dumps({"message": "There is no prompt in body"}), 400)
    
    prompt = body['prompt']

    if "width" in body.keys():
        w = body["width"]
    else:
        w = 512

    if "height" in body.keys():
        h = body["height"]
    else:
        h = 512

    image = SDApi.text2img(prompt, h, w)
    return Response(json.dumps({"image": image}), 200)


@app.post("/depth2img")
async def depth2img(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "prompt" not in body.keys():
        return Response(json.dumps({"message": "There is no prompt in body"}), 400)
    
    if "image" not in body.keys():
        return Response(json.dumps({"message": "There is no image in body"}), 400)
    
    prompt = body['prompt']

    image = body['image']

    if "width" in body.keys():
        w = body["width"]
    else:
        w = 512

    if "height" in body.keys():
        h = body["height"]
    else:
        h = 512

    image = SDApi.depth2img(image, prompt, h, w)
    return Response(json.dumps({"image": image}), 200)


@app.post("/inpaint")
async def inpaint(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "init_image" not in body.keys():
        return Response(json.dumps({"message": "There is no init image in body"}), 400)
    
    if "mask_image" not in body.keys():
        return Response(json.dumps({"message": "There is no mask image in body"}), 400)
    
    if "prompt" not in body.keys():
        return Response(json.dumps({"message": "There is no prompt in body"}), 400)
    
    prompt = body['prompt']

    init_image = body['init_image']

    mask_image = body["mask_image"]

    if "width" in body.keys():
        w = body["width"]
    else:
        w = 512

    if "height" in body.keys():
        h = body["height"]
    else:
        h = 512

    image = SDApi.inpaint(prompt, init_image, mask_image, h, w)
    return Response(json.dumps({"image": image}), 200)


@app.post("/imagemixer")
async def image_mixer(request: Request):

    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "init_image" not in body.keys():
        return Response(json.dumps({"message": "There is no init image in body"}), 400)
    
    if "style_image" not in body.keys():
        return Response(json.dumps({"message": "There is no style image in body"}), 400)
    

    init_image = body['init_image']

    style_image = body["style_image"]

    if "width" in body.keys():
        w = body["width"]
    else:
        w = 512

    if "height" in body.keys():
        h = body["height"]
    else:
        h = 512

    image = SDApi.image_mixer(init_image, style_image, h, w)
    return Response(json.dumps({"image": image}), 200)


@app.post("/rembg")
async def rembg(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    
    if "image" not in body.keys():
        return Response(json.dumps({"message": "There is no image in body"}), 400)
    

    image = body['image']


    image = SDApi.remove_background(image)
    return Response(json.dumps({"image": image}), 200)
    

@app.post("/magicmix")
async def magic_mix(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "prompt" not in body.keys():
        return Response(json.dumps({"message": "There is no prompt in body"}), 400)
    
    if "image" not in body.keys():
        return Response(json.dumps({"message": "There is no image in body"}), 400)
    
    prompt = body['prompt']

    image = body['image']

    if "width" in body.keys():
        w = body["width"]
    else:
        w = 512

    if "height" in body.keys():
        h = body["height"]
    else:
        h = 512

    image = SDApi.magic_mix(image, prompt, h, w)
    return Response(json.dumps({"image": image}), 200)


@app.post("/outpainting")
async def outpainting(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    
    if "image" not in body.keys():
        return Response(json.dumps({"message": "There is no image in body"}), 400)
    
    if "direction" not in body.keys():
        direction = "horizontal"
    else:
        direction = body["direction"]

    if "prompt" not in body.keys():
        prompt = ""
    else:
        prompt = body["prompt"]
    

    image = body['image']


    image = SDApi.outpainting(image, direction, prompt)
    return Response(json.dumps({"image": image}), 200)


@app.post("/facegen")
async def face_gen(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "prompt" not in body.keys():
        return Response(json.dumps({"message": "There is no prompt in body"}), 400)
    
    if "image" not in body.keys():
        return Response(json.dumps({"message": "There is no image in body"}), 400)
    
    prompt = body['prompt']

    image = body['image']

    if "width" in body.keys():
        w = body["width"]
    else:
        w = 512

    if "height" in body.keys():
        h = body["height"]
    else:
        h = 512

    image = SDApi.face_gen(image, prompt, h, w)
    return Response(json.dumps({"image": image}), 200)


@app.post("/upscale/set_task")
async def set_task(request: Request):
    headers = request.headers

    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)

    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    

    if "file_id" not in body.keys():
        return Response(json.dumps({"message": "No file id"}), 400)
    
    found_file = u.find_file(body["file_id"], files_dir_path + api_token_ + "/")
 
    if found_file is None:
        return Response(json.dumps({"message": "No file found"}), 400)
    
    target_file = body["file_id"] + found_file

    subprocess_data = {
        "-i" : files_dir_path + api_token_ + "/" + target_file,
        "-o" : output_files_path + api_token_ + "/" + target_file,
        "-s" : "2" if "scale" not in body.keys() else body["scale"] if int(body["scale"]) <= 4 else "2",
        "-n" : "realesrgan-x4plus" if "model" not in body.keys() else body["model"] if body["model"] in models_list else "realesrgan-x4plus",
    }

    cmd = ""
    os_name = os.name

    if os_name == "nt":
        cmd += "./utils/upscayl-bin.exe "
    else:
        cmd += os.path.abspath("utils/upscayl-bin") + " "

    for key, value in subprocess_data.items():
        cmd += f"{key} {value} "

    print(cmd)

    u.run_cmd(cmd)

    return Response(json.dumps({"message": "Success"}), 200)


@app.post("/upscale/upload_file")
async def upload_file(request: Request):

    headers = dict(request.headers)

    api_token_ = str(headers.get("api-token"))

    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)
    
    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "image" not in body.keys():
        return Response(json.dumps({"message": "There is no image in body"}), 400)

    extension = ".png"
    
    unique_id = str(uuid.uuid4())
    unique_filename = unique_id + extension

    os.makedirs(files_dir_path + api_token_ + "/", exist_ok=True)
    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    file_location = files_dir_path + api_token_ + "/" +  unique_filename


    cv2.imwrite(file_location, ip.b64_cv_im(body["image"]))

    return Response(json.dumps({"id": unique_id}), 200)


@app.post("/upscale/get_file")
async def get_file(request: Request, background_tasks: BackgroundTasks):
    headers = dict(request.headers)


    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)
    
    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "file_id" not in body.keys():
        return Response(json.dumps({"message": "No file id"}), 400)
    
    found_file = u.find_file(body["file_id"], files_dir_path)

    if found_file is None:
        return Response(json.dumps({"message": "No file found"}), 400)
    
    target_file = body["file_id"] + found_file
    
    file_path = output_files_path + api_token_ + "/" +  target_file
    input_file_path = files_dir_path + api_token_ + "/" +  target_file

    if not os.path.exists(file_path):
        return Response(json.dumps({"message": "File processing doesn't done, please wait"}), 400)
    
    time.sleep(3)

    background_tasks.add_task(remove_file_delay, file_path, delay = del_delay)
    background_tasks.add_task(remove_file_delay, input_file_path, delay = del_delay)

    return Response(json.dumps({"image": ip.cv_b64_im(cv2.imread(file_path))}), 200)

