##mask_generator.py##
import numpy as np
import cv2

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

class PointCloudGenerator:
    @staticmethod
    def create_class_point_cloud(json_data_list, image_shape, alpha=0.1):
        """여러 이미지의 클래스별 포인트 클라우드 생성"""
        # 클래스별 포인트 클라우드를 저장할 딕셔너리
        class_clouds = {class_name: np.zeros((*image_shape[:2], 3), dtype=np.float32) 
                       for class_name in CLASSES}
        
        # 모든 JSON 데이터에 대해 처리
        for json_data in json_data_list:
            for annotation in json_data['annotations']:
                points = np.array(annotation['points'])
                class_name = annotation['label']
                class_idx = CLASSES.index(class_name)
                
                # 임시 마스크 생성
                temp_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(temp_mask, [points.astype(np.int32)], 1)
                
                # 해당 클래스의 포인트 클라우드에 추가
                for i in range(3):
                    class_clouds[class_name][:, :, i][temp_mask == 1] += PALETTE[class_idx][i] * alpha
        
        # 각 클래스별 포인트 클라우드 정규화
        for class_name in CLASSES:
            class_clouds[class_name] = np.clip(class_clouds[class_name], 0, 255).astype(np.uint8)
            
        return class_clouds