import os
import http.server
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import json
import urllib.parse as urlparse

from inference_manager import InferenceManager

class Resquest(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def getQuery(self):
        print(self.path)
        o = urlparse.urlparse(self.path)
        q = urlparse.parse_qs(o.query)
        uploadedFilename =q['uploadedFilename'][0]
        fileFromDB = q['fileFromDB'][0]
        croppedFileNumber = q['croppedFileNumber'][0]
        return uploadedFilename, fileFromDB, croppedFileNumber

    def do_GET(self):
        print('>>>>>> Do get')
        self._set_headers()
        uploadedFilename, fileFromDB, croppedFileNumber = self.getQuery()
        print(uploadedFilename)
        print(fileFromDB)
        print(croppedFileNumber)
        # Get inference result
        result_base64 = inferenceManager.infer(uploadedFilename, fileFromDB, croppedFileNumber)
        data = {
            'result_base64': result_base64.decode("utf-8")
            }  # deco:bytes2str
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        print('>>>>>> Do post')
        self._set_headers()
        self.data_string = self.rfile.read(int(self.headers['Content-Length']))
        data = json.loads(self.data_string)['params']
        uploadedFilename = data['uploadedFilename']
        fileFromDB = data['fileFromDB']
        croppedFileNumber = data['croppedFileNumber']
        print(uploadedFilename)
        print(fileFromDB)
        print(croppedFileNumber)
        # Get inference result
        result_base64 = inferenceManager.infer(uploadedFilename, fileFromDB, croppedFileNumber)
        
        data = {
            'result_base64': result_base64.decode("utf-8")
            }  # deco:bytes2str
        
        self.wfile.write(json.dumps(data).encode())

if __name__ == "__main__":
    # Define the http server setting
    PORT = 8888

    # Define the model config
    cfg_file_list = [
        'configs/few_shot/e2e_mask_rcnn_R-50-FPN_1x_1.yaml',
        'configs/few_shot/e2e_mask_rcnn_R-50-FPN_1x_1.yaml', 
        'configs/few_shot/e2e_mask_rcnn_R-50-FPN_1x_2.yaml',
        'configs/few_shot/e2e_mask_rcnn_R-50-FPN_1x_3.yaml',
        'configs/few_shot/e2e_mask_rcnn_R-50-FPN_1x_4.yaml',
        'configs/few_shot/e2e_mask_rcnn_R-50-FPN_1x_5.yaml'
        ]

    load_ckpt_list = [
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_1_1shot/Jun27-21-46-32_gn1217.twcc.ai_step/ckpt/model_1shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_1_1shot/Jun27-21-46-32_gn1217.twcc.ai_step/ckpt/model_1shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_2_1shot/Jun27-21-48-10_gn1116.twcc.ai_step/ckpt/model_1shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_3_1shot/Jun27-21-48-12_gn1116.twcc.ai_step/ckpt/model_1shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_4_1shot/Jun27-21-48-24_gn1215.twcc.ai_step/ckpt/model_1shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_5_1shot/Jun27-21-49-03_gn1116.twcc.ai_step/ckpt/model_1shot_step19999.pth",]
    
    load_ckpt_list_5shot = [
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_1_5shot/Jun28-10-35-18_gn1117.twcc.ai_step/ckpt/model_5shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_1_5shot/Jun28-10-35-18_gn1117.twcc.ai_step/ckpt/model_5shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_2_5shot/Jun28-10-40-55_gn1217.twcc.ai_step/ckpt/model_5shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_3_5shot/Jun28-10-40-39_gn1117.twcc.ai_step/ckpt/model_5shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_4_5shot/Jun28-10-43-32_gn1117.twcc.ai_step/ckpt/model_5shot_step19999.pth",
        "../Outputs_aug/e2e_mask_rcnn_R-50-FPN_1x_5_5shot/Jun28-10-43-23_gn1230.twcc.ai_step/ckpt/model_5shot_step19999.pth",
    ]
    
    load_ckpt_list_deform = [
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_1_1shot/Jul09-00-27-13_gn1204.twcc.ai_step/ckpt/model_1shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_1_1shot/Jul09-00-27-13_gn1204.twcc.ai_step/ckpt/model_1shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_2_1shot/Jul09-00-27-20_gn1101.twcc.ai_step/ckpt/model_1shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_3_1shot/Jul09-00-27-39_gn1104.twcc.ai_step/ckpt/model_1shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_4_1shot/Jul09-00-27-42_gn1130.twcc.ai_step/ckpt/model_1shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_5_1shot/Jul09-00-27-51_gn1130.twcc.ai_step/ckpt/model_1shot_step39999.pth",
    ]

    load_ckpt_list_5shot_deform = [
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_1_5shot/Jul09-19-54-44_gn1204.twcc.ai_step/ckpt/model_5shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_1_5shot/Jul09-19-54-44_gn1204.twcc.ai_step/ckpt/model_5shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_2_5shot/Jul09-19-56-12_gn1204.twcc.ai_step/ckpt/model_5shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_3_5shot/Jul09-19-56-23_gn1204.twcc.ai_step/ckpt/model_5shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_4_5shot/Jul09-19-56-33_gn1204.twcc.ai_step/ckpt/model_5shot_step39999.pth",
        "../Outputs_deform/e2e_mask_rcnn_R-50-FPN_1x_5_5shot/Jul09-19-56-43_gn1204.twcc.ai_step/ckpt/model_5shot_step39999.pth",
    ]
    
    database_dir = '../cfisDemo/public/database/'
    upload_dir = '../cfisDemo/public/uploaded_imgs/'
    result_dir = '../cfisDemo/public/results/'

    # Initialize the model manager
    inferenceManager = InferenceManager(cfg_file_list, load_ckpt_list, load_ckpt_list_5shot, database_dir, upload_dir, result_dir)

    # Start the http server
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Resquest) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()