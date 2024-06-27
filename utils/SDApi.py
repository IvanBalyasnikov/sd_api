import numpy as np
import requests
import base64
import cv2



class SDApi():

    def __init__(self, ip:str = "localhost", port:str = "8000", url:str = None) -> None:
        
        if url != None:
            self.url = url
        else:
            self.url = "http://" + ip + ":" + port


    def cv_b64_im(self, img):
        retval, bytes = cv2.imencode('.png', img)
        return base64.b64encode(bytes).decode('utf-8')


    def b64_cv_im(self, img_str):

        im_bytes = base64.b64decode(img_str)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  
        return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    def depth2img(self, depth_img, prompt, w,h):

        url = self.url

        img = depth_img

        payload = {
            "prompt": prompt,
            'negative_prompt': "EasyNegative",
            "steps": 20,
            "width": w,
            "height": h,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                        "input_image": img,
                        "module": "depth_midas",
                        "model": "control_sd15_depth [fef5e48e]"
                        }
                    ]
                }
            },
        }


        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        r = response.json()

        return r['images'][0]


    def image_mixer(self, img1, img2, h,w):

        url = self.url

        payload = {
            'negative_prompt': "EasyNegative",
            "steps": 20,
            "width": w,
            "height": h,
            "init_images": [
                img2
            ],
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                        "input_image": img1,
                        "module": "canny",
                        "model": "control_sd15_canny [fef5e48e]"
                        }
                    ]
                }
            },
        }

        response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
        r = response.json()

        return r['images'][0]


    def remove_background(self, img):

        url = self.url

        payload = {
            "input_image": img,
            "model": "u2net",
            "return_mask": False,
            "alpha_matting": False,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 10
        }
            
        response = requests.post(url=f'{url}/rembg', json=payload)
        r = response.json()

        return r['image']


    def magic_mix(self, img1, prompt, h,w):

        url = self.url

        payload = {
            "prompt": prompt,
            'negative_prompt': "EasyNegative",
            "steps": 20,
            "width": w,
            "height": h,
            "init_images": [
                img1
            ],
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                        "input_image": img1,
                        "module": "canny",
                        "model": "control_sd15_canny [fef5e48e]"
                        }
                    ]
                }
            },
        }

        response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
        r = response.json()

        print(r)

        return r['images'][0]


    def outpainting(self, img1, direction, prompt):

        url = self.url

        image_cv = self.b64_cv_im(img1)
        h, w, _ = image_cv.shape
        

        payload = {
            "prompt":prompt,
            'negative_prompt': "EasyNegative",
            "steps": 20,
            "width": w + 128 if direction == "horizontal" else w,
            "height": h + 128 if direction == "vertical" else h,
            "resize_mode":1,
            "init_images": [
                img1
            ],
            # "script_name": "poor man's outpainting",
            # "script_args":[128, 4, "fill", ['left', 'right', 'up', 'down']]
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                        "image": img1,
                        "module": "inpaint_only+lama",
                        "model": "control_v11p_sd15_inpaint [ebff9138]",
                        "control_mode":'My prompt is more important',
                        "resize_mode":'Resize and Fill'
                        }
                    ]
                }
            },
        }

        response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
        r = response.json()

        return r['images'][0]


    def face_gen(self, img1, prompt, h, w):

        url = self.url

        payload = {
            "prompt": prompt + ", <lora:ip-adapter-faceid-plus_sd15_lora:1>",
            'negative_prompt': "EasyNegative",
            "steps": 20,
            "width": w,
            "height": h,
            "init_images": [
                img1
            ],
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                        "input_image": img1,
                        "module": "ip-adapter_face_id_plus",
                        "model": "ip-adapter-faceid-plus_sd15 [d86a490f]"
                        }
                    ]
                }
            },
        }

        response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
        r = response.json()
        
        return r['images'][0]
    
    def text2img(self, prompt, h, w):

        url = self.url

        payload = {
            "prompt": prompt,
            'negative_prompt': "EasyNegative",
            "steps": 20,
            "width": w,
            "height": h,
        }

        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        r = response.json()
        
        return r['images'][0]
    

    def inpaint(self, prompt, img1, mask, h,w):

        url = self.url

        payload = {
            "prompt": prompt,
            'negative_prompt': "EasyNegative",
            "steps": 20,
            "width": w,
            "height": h,
            "init_images": [
                img1
            ],
            "mask": mask
        }

        response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
        r = response.json()

        return r['images'][0]






