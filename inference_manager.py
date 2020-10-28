import os
import glob
from tools.infer_few_shot import initialize, reload_ckpt, segment

DATASETS = {
    1: "BBBC039",
    2: "ClusterNuclei",

    3: "BBBC020_nuclei",
    4: "BBBC020_cell",
    5: "TNBC",

    6: "MicroNet",
    7: "BBBC018_nuclei",
    8: "BBBC018_cell",

    9: "NucleiSeg",
    10: "MoNuSeg",

    11: "ISBI2009",
    12: "BBBC007_nuclei",
    13: "BBBC007_cell",
    14: "Hela"
}

INV_DATASETS = {v: k for k, v in DATASETS.items()}

TEST_SPLIT = {
    1: 1,
    2: 1,

    3: 2,
    4: 2,
    5: 2,

    6: 3,
    7: 3,
    8: 3,

    9: 4,
    10: 4,

    11: 5,
    12: 5,
    13: 5,
    14: 5
}

class InferenceManager():
    def __init__(self, cfg_file_list, load_ckpt_list, load_ckpt_list_5shot, database_dir, upload_dir, result_dir, deform_conv=False):
        self.cfg_file_list = cfg_file_list
        self.load_ckpt_list = load_ckpt_list
        self.load_ckpt_list_5shot = load_ckpt_list_5shot
        self.current_ckpt = load_ckpt_list[0]
        self.database_dir = database_dir
        self.upload_dir = upload_dir
        self.result_dir = result_dir
        self.deform_conv = deform_conv
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        #self.database_img_list = sorted(glob.glob(os.path.join(self.database_dir, '*')))
        self.model = initialize(self.cfg_file_list[0], self.load_ckpt_list[0], self.deform_conv)

    def infer(self, img_name, is_from_db, n_shot):
        group_name = os.path.splitext(img_name)[0]
        query_img_list = []
        if is_from_db:
            ext = os.path.splitext(img_name)[-1]
            img_name = '_'.join(img_name.split('_')[:-1]) + ext
            print(img_name)
            target_img_name = os.path.join(self.database_dir, img_name) 
            query_img_list = sorted(glob.glob(os.path.join(self.upload_dir, group_name, '*')))
            category_name = '_'.join(group_name.split('_')[1:]).split('-')[0]
            split = TEST_SPLIT[INV_DATASETS[category_name]]
            assert len(query_img_list) == n_shot, 'the number of shots and the number of uploaded imgs is not the same!!!'
            if len(query_img_list) >= 5:
                if self.current_ckpt != self.load_ckpt_list_5shot[split]:
                    self.model = reload_ckpt(self.load_ckpt_list_5shot[split])
                    self.current_ckpt = self.load_ckpt_list_5shot[split]
            else:
                if self.current_ckpt != self.load_ckpt_list[split]:
                    self.model = reload_ckpt(self.load_ckpt_list[split])
                    self.current_ckpt = self.load_ckpt_list[split]
            result_base64 = segment(group_name, target_img_name, query_img_list, self.model, self.result_dir, category_name)

        else:
            target_img_name = os.path.join(self.upload_dir, group_name, img_name)
            query_img_list = sorted(set(glob.glob(os.path.join(self.upload_dir, group_name, '*')))\
                                    - set([target_img_name]))
            assert len(query_img_list) == n_shot, 'the number of shots and the number of uploaded imgs is not the same!!!'
            if len(query_img_list) >= 5:
                if self.current_ckpt != self.load_ckpt_list_5shot[0]:
                    self.model = reload_ckpt(self.load_ckpt_list_5shot[0])
                    self.current_ckpt = self.load_ckpt_list_5shot[0]
            else:
                if self.current_ckpt != self.load_ckpt_list[0]:
                    self.model = reload_ckpt(self.load_ckpt_list[0])
                    self.current_ckpt = self.load_ckpt_list[0]
            result_base64 = segment(group_name, target_img_name, query_img_list, self.model, self.result_dir)
        
        return result_base64