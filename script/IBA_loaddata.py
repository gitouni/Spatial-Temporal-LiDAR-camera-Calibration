import numpy as np
import argparse
import os
import cv2
import yaml
import json
import shutil
from collections import OrderedDict

os.chdir(os.path.dirname(__file__))

def str2bool(s:str) -> bool:
    if s.isdigit():
        if float(s) > 0:
            return True
        else:
            return False
    if s.lower() == "false":
        return False
    else:
        return True

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--keyframe_dir",type=str,default="../KITTI-00/KeyFrames/")
    io_parser.add_argument("--keyframe_id",type=str,default="../KITTI-00/FrameId.yml")
    io_parser.add_argument("--key_config",type=str,default="../config/debug/KeyFrameIO.yml")
    io_parser.add_argument("--save_dir",type=str,default="../debug/data/")
    io_parser.add_argument("--keyframe_index",type=int,nargs="+",default=[50*i for i in range(2)])
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--best_convis_num",type=int,default=1)
    args = parser.parse_args()
    return args


def get_fs_info(fs, keypt_nodename, mappt_nodename, corrkpt_nodename):
    kptnode = fs.getNode(keypt_nodename)
    keypts = [[int(kptnode.at(i).at(0).real()), int(kptnode.at(i).at(1).real())] for i in range(kptnode.size())]
    mapptnode = fs.getNode(mappt_nodename)
    mappt_indices = [int(mapptnode.at(i).real()) for i in range(mapptnode.size())]
    corrkpt_node = fs.getNode(corrkpt_nodename)
    corr_keypt_indices = [int(corrkpt_node.at(i).real()) for i in range(corrkpt_node.size())]
    return np.array(keypts), mappt_indices, corr_keypt_indices

def getMatchedId(src_mappt_indices:list, tgt_mappt_indices:list, src_corrkpt_indices:list, tgt_corrkpt_indices:list):
    src_matched_indices = []
    tgt_matched_indices = []
    for i in range(len(src_mappt_indices)):
        src_mappt_id = src_mappt_indices[i]
        if src_mappt_id in tgt_mappt_indices:
            tgt_id = tgt_mappt_indices.index(src_mappt_id)
            src_matched_indices.append(src_corrkpt_indices[i])
            tgt_matched_indices.append(tgt_corrkpt_indices[tgt_id])
    return np.array(src_matched_indices), np.array(tgt_matched_indices)  # Keypoint Indices
        

if __name__ == "__main__":
    args = options()
    KeyFramesFiles = list(sorted(os.listdir(args.keyframe_dir)))
    KeyFramesFiles = [file for file in KeyFramesFiles if os.path.splitext(file)[1] == '.yml']
    keyframe_indices:list = yaml.load(open(args.keyframe_id,'r'), yaml.SafeLoader)["mnId"]
    key_config = yaml.load(open(args.key_config,'r'), yaml.SafeLoader)
    attr_key = key_config["attribution"]
    connect_key = key_config["connect"]
    
    src_indices = args.keyframe_index
    cnt = 0
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    for src_index in src_indices:
        src_KeyFrameFile = os.path.join(args.keyframe_dir, KeyFramesFiles[src_index])
        src_fs = cv2.FileStorage(src_KeyFrameFile, cv2.FILE_STORAGE_READ)
        ordered_convis_node = src_fs.getNode(connect_key["keyframe"])
        src_keypts, src_mappt_id, src_corrkpt_id = get_fs_info(src_fs, attr_key["keypoint"], connect_key["mappoint"], connect_key["corr_keypoint"])
        src_pose = src_fs.getNode(attr_key["pose"]).mat()
        for i in range(min(args.best_convis_num, ordered_convis_node.size())):
            tgt_index = keyframe_indices.index(int(ordered_convis_node.at(i).real()))
            tgt_KeyFrameFile = os.path.join(args.keyframe_dir, KeyFramesFiles[tgt_index])
            tgt_fs = cv2.FileStorage(tgt_KeyFrameFile, cv2.FILE_STORAGE_READ)
            tgt_keypts, tgt_mappt_id, tgt_corrkpt_id = get_fs_info(tgt_fs, attr_key["keypoint"], connect_key["mappoint"], connect_key["corr_keypoint"])
            src_matched_pts_idx, tgt_matched_pts_idx = getMatchedId(src_mappt_id, tgt_mappt_id, src_corrkpt_id, tgt_corrkpt_id)
            src_matched_kpt = src_keypts[src_matched_pts_idx]
            tgt_matched_kpt = tgt_keypts[tgt_matched_pts_idx]
            tgt_pose = tgt_fs.getNode(attr_key["pose"]).mat()
            write_data = OrderedDict()
            write_data["src_index"] = src_index
            write_data["tgt_index"] = tgt_index
            write_data["src_mnId"] = int(src_fs.getNode(attr_key["id"]).real())
            write_data["tgt_mnId"] = int(tgt_fs.getNode(attr_key["id"]).real())
            write_data["src_mnFrameId"] = int(src_fs.getNode(attr_key["frame_id"]).real())
            write_data["tgt_mnFrameId"] = int(tgt_fs.getNode(attr_key["frame_id"]).real())
            write_data["src_pose"] = src_pose.tolist()
            write_data["tgt_pose"] = tgt_pose.tolist()
            write_data["src_matched_keypoint"] = src_matched_kpt.tolist()
            write_data["tgt_matched_keypoint"] = tgt_matched_kpt.tolist()
            json.dump(write_data, open(os.path.join(args.save_dir, "%06d.json"%cnt),'w'),indent=4)
            cnt += 1
    