# import zipfile
# import os

# zip_file_path = '/home/Student/s4819764/classification.zip'
# destination_path = '/home/Student/s4819764/database'

# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(destination_path)

import os
import shutil
import numpy as np

def create_test_set(source_dir, test_dir, test_size=0.10):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            test_category_path = os.path.join(test_dir, category)
            if not os.path.exists(test_category_path):
                os.makedirs(test_category_path)

            files = [file for file in os.listdir(category_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
            np.random.shuffle(files)

            n_test = int(len(files) * test_size)
            test_files = files[:n_test]

            for file in test_files:
                shutil.move(os.path.join(category_path, file), os.path.join(test_category_path, file))
create_test_set('/home/Student/s4819764/database/carabid', '/home/Student/s4819764/database/carabid_test', test_size=0.10)

# import os
# import shutil

# def merge_folders(source_folder, target_folder):
#     """
#     递归地将source_folder中的所有文件和子文件夹复制到target_folder中。
#     保持文件结构不变，并为重复的文件名添加数字后缀以避免覆盖。
#     """
#     for item in os.listdir(source_folder):
#         source_path = os.path.join(source_folder, item)
#         target_path = os.path.join(target_folder, item)

#         if os.path.isdir(source_path):
#             # 如果是文件夹
#             os.makedirs(target_path, exist_ok=True)
#             merge_folders(source_path, target_path)
#         else:
#             # 如果是文件
#             if os.path.exists(target_path):
#                 # 文件名冲突
#                 base, extension = os.path.splitext(target_path)
#                 counter = 1
#                 new_target_path = f"{base}_{counter}{extension}"
#                 while os.path.exists(new_target_path):
#                     counter += 1
#                     new_target_path = f"{base}_{counter}{extension}"
#                 target_path = new_target_path

#             shutil.copy2(source_path, target_path)

# source_folders = ['/home/Student/s4819764/database/classification/train', '/home/Student/s4819764/database/classification/val']
# target_folder = '/home/Student/s4819764/database/IP102'

# os.makedirs(target_folder, exist_ok=True)

# for folder in source_folders:
#     merge_folders(folder, target_folder)

# print("Folders have been merged successfully.")