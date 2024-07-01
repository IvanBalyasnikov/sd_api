from fastapi import FastAPI, Request, Response, BackgroundTasks
from configparser import ConfigParser

import utils.image_processing as ip
import utils.SDApi as sda 

import subprocess
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


def find_file(file_name, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(file_name):
                return os.path.splitext(file)[1]
    return None


def run_cmd(cmd: str):
    # subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    subprocess.Popen(cmd)


def run_task(func, *args):
    print(func)
    image = func(*args[:-2])
    cv2.imwrite(args[-2], ip.b64_cv_im(image))
    args[-1].add_task(remove_file_delay, args[-2], delay = del_delay)



@app.post("/txt2img")
async def txt2img(request: Request, background_tasks: BackgroundTasks):
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


    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.text2img, prompt, h, w, file_path, background_tasks)


    return Response(json.dumps({"job_id": unique_id}), 200)


@app.post("/depth2img")
async def depth2img(request: Request, background_tasks: BackgroundTasks):
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

    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.depth2img, image, prompt, h, w, file_path, background_tasks)


    return Response(json.dumps({"job_id": unique_id}), 200)


@app.post("/inpaint")
async def inpaint(request: Request, background_tasks: BackgroundTasks):
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

    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.inpaint, prompt, init_image, mask_image, h, w, file_path, background_tasks)


    return Response(json.dumps({"job_id": unique_id}), 200)



@app.post("/imagemixer")
async def image_mixer(request: Request, background_tasks: BackgroundTasks):

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

    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.image_mixer, init_image, style_image, h, w, file_path, background_tasks)


    return Response(json.dumps({"job_id": unique_id}), 200)


@app.post("/rembg")
async def rembg(request: Request, background_tasks: BackgroundTasks):
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

    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.remove_background, image, file_path, background_tasks)

    return Response(json.dumps({"job_id": unique_id}), 200)

@app.post("/magicmix")
async def magic_mix(request: Request, background_tasks: BackgroundTasks):
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

    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.magic_mix, image, prompt, h, w, file_path, background_tasks)

    return Response(json.dumps({"job_id": unique_id}), 200)


@app.post("/outpainting")
async def outpainting(request: Request, background_tasks: BackgroundTasks):
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


    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.outpainting, image, direction, prompt, file_path, background_tasks)

    return Response(json.dumps({"job_id": unique_id}), 200)



@app.post("/facegen")
async def face_gen(request: Request, background_tasks: BackgroundTasks):
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

    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"

    background_tasks.add_task(run_task, SDApi.face_gen, image, prompt, h, w, file_path, background_tasks)

    return Response(json.dumps({"job_id": unique_id}), 200)


@app.post("/upscale")
async def upscale(request: Request, background_tasks: BackgroundTasks):
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
    
    os.makedirs(output_files_path + api_token_ + "/", exist_ok=True)
    os.makedirs(files_dir_path + api_token_ + "/", exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    file_path =  output_files_path + api_token_ + "/" + unique_id + ".png"
    target_file =  files_dir_path + api_token_ + "/" + unique_id + ".png"
    cv2.imwrite(target_file, ip.b64_cv_im(body['image']))


    subprocess_data = {
        "-i" : target_file,
        "-o" : file_path,
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

    run_cmd(cmd)

    background_tasks.add_task(remove_file_delay, file_path, delay = del_delay)
    background_tasks.add_task(remove_file_delay, target_file, delay = del_delay)

    return Response(json.dumps({"job_id": unique_id}), 200)



@app.post("/get_file")
async def get_file(request: Request):
    headers = dict(request.headers)


    api_token_ = headers.get("api-token")
    if api_token_ not in api_tokens:
        return Response(json.dumps({"message": "Invalid api token"}), 403)
    
    body_bytes = await request.body()
    try:
        body = dict(json.loads(body_bytes))
    except json.JSONDecodeError:
        return Response(json.dumps({"message": "Could not decode body as JSON"}), 400)
    
    if "job_id" not in body.keys():
        return Response(json.dumps({"message": "No job id"}), 400)
    
    if not os.path.isfile(output_files_path + api_token_ + "/" +  body["job_id"] + ".png"):
        return Response(json.dumps({"message": "File processing doesn't done or save file time is expired, please wait or retry some method"}), 400)


    file_path = output_files_path + api_token_ + "/" +  body["job_id"] + ".png"

    time.sleep(2)

    return Response(json.dumps({"image": ip.cv_b64_im(cv2.imread(file_path))}), 200)

