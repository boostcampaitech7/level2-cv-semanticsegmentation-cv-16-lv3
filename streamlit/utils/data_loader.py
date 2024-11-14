##data_loader.py##
import os
from PIL import Image
import json

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "train", "DCM")
        self.json_dir = os.path.join(data_dir, "train", "outputs_json")
    
    def get_image_list(self):
        image_files = []
        print(f"Searching in directory: {self.images_dir}")  # 검색 디렉토리 출력
        
        for root, dirs, files in os.walk(self.images_dir):
            print(f"Current directory: {root}")  # 현재 검색 중인 디렉토리
            print(f"Found directories: {dirs}")  # 발견된 하위 디렉토리들
            print(f"Found files: {files}")  # 발견된 파일들
            
            for f in files:
                if f.endswith('.png'):
                    rel_path = os.path.relpath(root, self.images_dir)
                    full_path = os.path.join(rel_path, f)
                    image_files.append(full_path)
                    print(f"Added image: {full_path}")  # 추가된 이미지 경로
        
        return sorted(image_files)
    
    def load_image(self, image_path):
        full_path = os.path.join(self.images_dir, image_path)
        return Image.open(full_path)
    def get_json_path(self, image_path):
        image_dir, image_name = os.path.split(image_path)
        json_name = image_name.replace('.jpg', '.json').replace('.png', '.json')
        return os.path.join(self.json_dir, image_dir, json_name)
    
    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def get_image_pairs(self, image_files):
        """폴더별로 이미지 쌍을 찾아서 반환하는 함수"""
        pairs = {}
        for file in image_files:
            # 파일의 디렉토리 경로와 파일명을 분리
            dir_path = os.path.dirname(file)
            
            # 같은 폴더에 있는 파일들을 쌍으로 묶음
            if dir_path not in pairs:
                pairs[dir_path] = {'L': None, 'R': None}
            
            # 파일명에 따라 L/R 구분
            # 예시: 첫 번째 파일을 L, 두 번째 파일을 R로 지정
            if pairs[dir_path]['L'] is None:
                pairs[dir_path]['L'] = file
            else:
                pairs[dir_path]['R'] = file
                
        # L과 R 이미지가 모두 있는 쌍만 반환
        return {k: v for k, v in pairs.items() if v['L'] is not None and v['R'] is not None}