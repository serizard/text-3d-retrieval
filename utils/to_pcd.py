# © Copyright (c) 2024  Semin Kim & Dohyeong Han
# © 2024 [Semin & Dohyeong]. All Rights Reserved.
# © 2024 [Semin & Dohyeong]. All rights reserved. Unauthorized duplication or distribution is prohibited.

import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# .off 파일을 읽어 정점 및 면 데이터를 추출하는 함수!
def read_off(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # 첫 줄에 "OFF" 문자열이 있어야 함
        if lines[0].strip() != "OFF":
            raise ValueError("Not a valid OFF file")
        
        # 두 번째 줄에 정점 수, 면 수, 에지 수가 있어야 함
        parts = lines[1].strip().split()
        num_vertices = int(parts[0])
        num_faces = int(parts[1])
        
        # 정점 좌표 읽기
        vertices = []
        for i in range(2, 2 + num_vertices):
            vertex = list(map(float, lines[i].strip().split()))
            vertices.append(vertex)
        
        # 면 데이터 읽기
        faces = []
        for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
            face = list(map(int, lines[i].strip().split()[1:]))  # 첫 번째 숫자는 정점의 수이므로 제외!!
            faces.append(face)
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        print(f"Read OFF file: {file_path}")
        print(f"Number of vertices: {vertices.shape[0]}")
        print(f"Number of faces: {faces.shape[0]}")
        
        return vertices, faces

# .off 파일을 .npy 파일로 변환하는 함수!
def convert_off_to_npy(off_file_path, npy_file_path):
    vertices, faces = read_off(off_file_path)
    data = {'vertices': vertices, 'faces': faces}
    np.save(npy_file_path, data, allow_pickle=True)
    print(f"Converted {off_file_path} to {npy_file_path}")
    print('hieqreqreqwr')

# 파일 선택 및 변환 실행 함수!
def select_and_convert_file():
    # 파일 선택 대화 상자 열기
    off_file_path = filedialog.askopenfilename(filetypes=[("OFF files", "*.off")])
    if not off_file_path:
        print("No file selected")
        return

    # 저장할 .npy 파일 경로 설정
    npy_file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NPY files", "*.npy")])
    if not npy_file_path:
        print("Save operation cancelled")
        return

    # 변환 실행
    convert_off_to_npy(off_file_path, npy_file_path)

# .npy 파일을 로드하여 3D 데이터를 시각화하는 함수!
def load_and_plot_npy(npy_file_path):
    data = np.load(npy_file_path, allow_pickle=True).item()
    vertices = data['vertices']
    faces = data['faces']

    # 점의 개수 및 면의 개수 출력
    num_points = vertices.shape[0]
    num_faces = faces.shape[0]
    print(f"The .npy file contains {num_points} points and {num_faces} faces.")

    # 3D 시각화
    fig = plt.figure(num=f"Number of Points: {num_points}")
    ax = fig.add_subplot(111, projection='3d')

    # 정점 좌표 추출
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # 정점 설정
    ax.scatter(x, y, z, c='b', marker='o', s=1)  # 정점 크기를 최소로 설정

    # 면 설정
    poly3d = [[vertices[idx] for idx in face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.1,  edgecolors='r', alpha=.25)) 

    # 축 설정
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 각 축의 범위 계산
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    plt.show()

# 파일 선택 및 시각화 실행 함수!
def select_and_view_file():
    # 파일 선택 대화 상자 열기
    npy_file_path = filedialog.askopenfilename(filetypes=[("NPY files", "*.npy")])
    if not npy_file_path:
        print("No file selected")
        return

    # 3D 시각화 실행
    load_and_plot_npy(npy_file_path)

# Tkinter 창 숨기기
root = tk.Tk()
root.withdraw()

# 변환 및 시각화 메뉴
def main_menu():
    while True:
        print("Select an option:")
        print("1. Convert OFF to NPY")
        print("2. View NPY file")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            select_and_convert_file()
        elif choice == '2':
            select_and_view_file()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

# ★★★★★ 실행 ★★★★★
main_menu()
