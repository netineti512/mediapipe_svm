# 拡張子名を統一する

今回は画像
imageフォルダに画像を入れてください
```convert.py
import os
from pathlib import Path

directory = './image'

for f in Path(directory).rglob('*'):
   f.rename(f.stem + '.JPG')
```
# 画像の名前を連番にする。

例)
fuck001.jpg silent001.jpg
...
fuck100.jpg silent100.jpg
```rename.py
import os

list = ["fuck", "good", "silent", "up"]
num = [1, 2, 3, 4]
#list = [1, 2, 3, 4]

for k in list:
    for j in range(1,301):
        # 変更前ファイル
        path1 = f'./Oimage/{k}/{k}{j}.JPG'
                
        # 変更後ファイル
        path2= f'./image/{k}/{j}.JPG'
                
        # ファイル名の変更 
        os.rename(path1, path2) 
                
        # ファイルの存在確認 
        print(os.path.exists(path2))
```
# [Python]画像上の手のmediapipeの処理を行う
[参考](https://qiita.com/h-ueno2/items/b8dd54b396add5c3b12a)
```mediapipe_hand.py
import cv2
import glob
import os
import csv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# CSVのヘッダを準備
fields = []
fields.append('file_name')
for i in range(21):
    fields.append(str(i)+'_x')
    fields.append(str(i)+'_y')
    fields.append(str(i)+'_z')

if __name__ == '__main__':
    # 元の画像ファイルの保存先を準備
    resource_dir = './image'
    # 対象画像の一覧を取得
    file_list = glob.glob(os.path.join(resource_dir, "*.*"))
    file_list.sort()

    # 保存先の用意
    save_csv_dir = './'
    os.makedirs(save_csv_dir, exist_ok=True)
    save_csv_name = './sample.csv'
    save_image_dir = './image_landmark/'
    os.makedirs(save_image_dir, exist_ok=True)

    with mp_hands.Hands(static_image_mode=True,
            max_num_hands=2, # 検出する手の数（最大2まで）
            min_detection_confidence=0.5) as hands, \
        open(os.path.join(save_csv_dir, save_csv_name), 
            'w', encoding='utf-8', newline="") as f:

        # csv writer の用意
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for file_path in file_list:
            # 画像の読み込み
            image = cv2.imread(file_path)

            # 鏡写しの状態で処理を行うため反転
            image = cv2.flip(image, 1)

            # OpenCVとMediaPipeでRGBの並びが違うため、
            # 処理前に変換しておく。
            # CV2:BGR → MediaPipe:RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 推論処理
            results = hands.process(image)

            # 前処理の変換を戻しておく。
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not results.multi_hand_landmarks:
                # 検出できなかった場合はcontinue
                continue

            # ランドマークの座標情報
            landmarks = results.multi_hand_landmarks[0]

            # CSVに書き込み
            record = {}
            record["file_name"] = os.path.basename(file_path)
            for i, landmark in enumerate(landmarks.landmark):
                record[str(i) + '_x'] = landmark.x
                record[str(i) + '_y'] = landmark.y
                record[str(i) + '_z'] = landmark.z
            writer.writerow(record)

            # 元画像上にランドマークを描画
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            # 画像を保存
            cv2.imwrite(
                os.path.join(save_image_dir, os.path.basename(file_path)),
                cv2.flip(image, 1))
```
# [Python]画像上の顔のmediapipeの処理を行う
[参考1](https://qiita.com/Esp-v2/items/ba48ea3b2491f3d6bbb4)
[参考2](https://google.github.io/mediapipe/solutions/face_mesh)
```mediapipe_facemesh.py
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import mediapipe as mp

# 初期設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def main():
    image = cv2.imread('./image')
    results = holistic.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # ランドマークの座標dataframeとarray_imageを取得
    df_xyz, landmark_image = landmark(image)

    # ランドマーク記載画像を整形、出力
    landmark_image = cv2.cvtColor(
        landmark_image, cv2.COLOR_BGR2RGB)  # BGRtoRGB
    landmark_image = Image.fromarray(landmark_image.astype(np.uint8))
    # landmark_image.show()

    height, width, channels = image.shape[:3]
    # ランドマークの色情報を取得
    df_rgb = color(image, df_xyz, height, width)

    # xyzとrgb結合
    df_xyz_rgb = pd.concat([df_xyz, df_rgb], axis=1)
    df_xyz_rgb.to_csv('./landmark.csv', header=False, index=False)

# ランドマークの画素値を取得する
def color(image, xyz, height, width):
    label = ['r', 'g', 'b']
    data = []
    for _ in range(len(xyz)):
        x = int(xyz.iloc[_, 0]*width)
        y = int(xyz.iloc[_, 1]*height)

        b = int(image[y, x, 0])
        g = int(image[y, x, 1])
        r = int(image[y, x, 2])

        data.append([r, g, b])

    df = pd.DataFrame(data, columns=label)
    return df

# ランドマークの座標を取得する
def face(results, annotated_image):
    label = ["x", "y", "z"]
    data = []
    if results.face_landmarks:
        # ランドマークを描画する
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for landmark in results.face_landmarks.landmark:
            data.append([landmark.x, landmark.y, landmark.z])

    else:  # 検出されなかったら欠損値nanを登録する
        data.append([np.nan, np.nan, np.nan])

    df = pd.DataFrame(data, columns=label)
    return df

# imageに対してmediapipeでランドマークを表示、出力する
def landmark(image):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()

    # ランドマーク取得
    df_xyz = face(results, annotated_image)
    return df_xyz, annotated_image

if __name__ == "__main__":
    main()
```

# [Python]リアルタイムで顔の座標をcsvに保存する
```facemesh_realtime.py
import cv2
import mediapipe as mp
import csv
import datetime

dt_now = datetime.datetime.now()
#num = dt_now.microsecond
num = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

time = []
time.append("time")

x = []
y = []
z = []
x.append("10_x")
y.append("10_y")
z.append("10_z")

body = []

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
        time.append(num)
        print(num)
        x.append(face_landmarks.landmark[10].x)
        y.append(face_landmarks.landmark[10].y)
        z.append(face_landmarks.landmark[10].z)
      num+=1
            
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

body.append(x)
body.append(y)
body.append(z)

with open("landmark.csv", "w", newline="") as f:
    w = csv.writer(f, delimiter=",")
    w.writerow(time)
    for data_list in body:
        w.writerow(data_list)
```

# MediaPipeでリアルタイムの向きの推定を行う
[参考](https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600)
```realtime_direction.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import datetime

dt_now = datetime.datetime.now()
#num = dt_now.microsecond
#num = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

body = []

time = []
num = 0
time.append("time")

direction = []
direction.append("direction")

x1 = []
x1.append("x")
y1 = []
y1.append("y")
z1 = []
z1.append("z")

while cap.isOpened():
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the results
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          for idx, lm in enumerate(face_landmarks.landmark):
              if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                  nose_2d = (lm.x * img_w, lm.y * img_h)
                  nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])

          # Convert it to the NumPy array
          face_2d = np.array(face_2d, dtype=np.float64)

          # Convert it to the NumPy array
          face_3d = np.array(face_3d, dtype=np.float64)

   	    # The camera matrix
          focal_length = 1 * img_w

          cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                  [0, focal_length, img_w / 2],
                                  [0, 0, 1]])

	    # The Distance Matrix
          dist_matrix = np.zeros((4, 1), dtype=np.float64)
	    # Solve PnP
          success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

	    # Get rotational matrix
          rmat, jac = cv2.Rodrigues(rot_vec)

	    # Get angles
          angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

	    # Get the y rotation degree
          x = angles[0] * 360
          y = angles[1] * 360
          z = angles[2] * 360
          time.append(num)
          print(num)
          x1.append(x)
          y1.append(y)
          z1.append(z)

          # See where the user's head tilting
          if y < -10:
              text = "Looking Left"
              direction.append(text)
              print(num, text, x, y)
          elif y > 10:
              text = "Looking Right"
              direction.append(text)
              print(num, text, x, y)
          elif x < -10:
              text = "Looking Down"
              direction.append(text)
              print(num, text, x, y)
          else:
              text = "Forward"
              direction.append(text)
              print(num, text, x, y)
          
          # Add the text on the image
          cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          
          # Display the nose direction
          nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

          p1 = (int(nose_2d[0]), int(nose_2d[1]))
          p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
          
          cv2.line(image, p1, p2, (255, 0, 0), 2)
			
          cv2.imshow('Head Pose Estimation', image)
        num += 1

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()

body.append(direction)
body.append(x1)
body.append(y1)
body.append(z1)

with open("landmark.csv", "w", newline="") as f:
  w = csv.writer(f, delimiter=",")
  w.writerow(time)
  for i in body:
    w.writerow(i)
```

# 顔と手CSVの座標データを結合させる
```hand21_face4.py
import csv
import pandas as pd 
import numpy as np

#list = ["fuck","good", "silent", "up"]
list = [1, 2, 3, 4]

for k in list:
    for j in range(1,301):
        print(k, j)

        #6と10と27と257
        #467箇所
        #行列を入れ替えた顔座標データを一次元で読み込む
        with open(f"./result1/{k}/landmark_face_{k}_{j}.csv") as file_name1:
            array1 = np.loadtxt(file_name1, delimiter=",").ravel()

        #x軸, y軸, z軸でそれぞれ分ける
        face_x = array1[::6]
        face_y = array1[1::6]
        face_z = array1[2::6]

        #点6,10,27,257取り出して格納
        face_x1 = []
        face_y1 = []
        face_z1 = []
        list = [6, 10, 27, 257]
        for i in list:
            face_x1.append(face_x[i])
            face_y1.append(face_y[i])
            face_z1.append(face_z[i])




        #手の座標データを開く
        with open(f"./result1/{k}/landmark_hand_{k}_{j}.csv") as file_name:
            array2 = np.loadtxt(file_name, delimiter=",", skiprows=1)
            #print(array)
        hand_x = array2[::3]
        hand_y = array2[1::3]
        hand_z = array2[2::3]

        #手の座標データに顔の座標データを結合させる
        x = np.append(hand_x, face_x1)
        y = np.append(hand_y, face_y1)
        z = np.append(hand_z, face_z1)

        #x, y, zをまとめる
        xyz = []
        list = [x, y, z]
        for i in list:
            xyz.extend(i)
        np.savetxt(f'./result2/{k}/landmark_hand21_face4_{k}_{j}.csv', xyz, delimiter=',')
```

# 手0からのすべての距離を統一する
```distance.py
import csv
import numpy as np
import math

#list = ["fuck","good", "silent", "up"]
list = [1, 2, 3, 4]

for k in list:
    for j in range(1,301):
        

        with open(f"./result2/{k}/landmark_hand21_face4_{k}_{j}.csv") as file_name:
            array = np.loadtxt(file_name, delimiter=",").ravel()
            #print(array)
            
        x1 = array[::3]
        y1 = array[1::3]
        #z1 = array[2::3]
            
        x2 = []
        y2 = []
        #z2 = []
        for i in range(0,25):
            print(k, j, i)
            a = x1[i] - x1[0]
            x2.append(a)
            b = y1[i] - y1[0]
            y2.append(b)
            #c = z1[i] - z1[0]
            #z2.append(c)
        #print(x2, y2, z2)

        l = math.sqrt((x2[1] - x2[0])**2 + (y2[1] - y2[0])**2)
        #l = math.sqrt((x2[1] - x2[0])**2 + (y2[1] - y2[0])**2 + (z2[1] - z2[0])**2)

        x = []
        y = []
        #z = []
        for i in range(24):
            d = x2[i] / l
            x.append(d)
            e = y2[i] / l
            y.append(e)
            #f = z2[i] / l
            #z.append(f)
        #print(x, y, z)

        xy = []
        xy.extend(x)
        xy.extend(y)
        np.savetxt(f'./result3/{k}/distance_hand_face_{k}_{j}.csv', xy, delimiter=',')

        """
        xyz = [] 
        xyz.extend(x)
        xyz.extend(y)
        xyz.extend(z)

        np.savetxt(f'./result3/{k}/distance_hand_face_{k}_{j}.csv', xyz, delimiter=',')
        """
```

# csvデータにファイル名を挿入する
```insert_filename.py
import csv
import pandas as pd 
import numpy as np

#list = ["fuck","good", "silent", "up"]
list = [1, 2, 3, 4]

for k in list:
    for j in range(1,301):
        print(k, j)

        with open(f"./result4/{k}/distance_hand_face_{k}_{j}.csv") as file_name1:
                    array1 = np.loadtxt(file_name1, delimiter=",").ravel()

        array2 = np.insert(array1, 0, j)
        array3 = np.insert(array2, 0, k)

        np.savetxt(f"./result5/{k}/distance_hand_face_{k}_{j}_fn.csv", array3, delimiter=',')
```

# csvファイルをtxtファイルに変更する
```csv_to_txt.py
import os
from pathlib import Path
list = [1, 2, 3, 4]

for k in list:
    directory = f'./../mediapipe/result4/{k}'

    for f in Path(directory).rglob('*'):
        f.rename("./txt/" + f.stem + '.txt')
```

# ラベルを付ける
```with_label.py
import numpy as np
import pandas as pd
import csv

csv = []
for i in range(1,5):
    for j  in range(1,301):
        txt = np.loadtxt(f"./txt/distance_hand_face_{i}_{j}_FileName.txt")
        df = pd.DataFrame(
                        {"pose":txt[0],
                        "filenumber":txt[1],
                        "X-hand0-hand1":txt[2],
                        "X-hand0-hand2":txt[3],
                        "X-hand0-hand3":txt[4],
                        "X-hand0-hand4":txt[5],
                        "X-hand0-hand5":txt[6],
                        "X-hand0-hand6":txt[7],
                        "X-hand0-hand7":txt[8],
                        "X-hand0-hand8":txt[9],
                        "X-hand0-hand9":txt[10],
                        "X-hand0-hand10":txt[11],
                        "X-hand0-hand11":txt[12],
                        "X-hand0-hand12":txt[13],
                        "X-hand0-hand13":txt[14],
                        "X-hand0-hand14":txt[15],
                        "X-hand0-hand15":txt[16],
                        "X-hand0-hand16":txt[17],
                        "X-hand0-hand17":txt[18],
                        "X-hand0-hand18":txt[19],
                        "X-hand0-hand19":txt[20],
                        "X-hand0-hand20":txt[21],
                        "X-hand0-face6":txt[22],
                        "X-hand0-face10":txt[23],
                        "X-hand0-face27":txt[24],
                        "X-hand0-face257":txt[25],
                        "Y-hand0-hand1":txt[26],
                        "Y-hand0-hand2":txt[27],
                        "Y-hand0-hand3":txt[28],
                        "Y-hand0-hand4":txt[29],
                        "Y-hand0-hand5":txt[30],
                        "Y-hand0-hand6":txt[31],
                        "Y-hand0-hand7":txt[32],
                        "Y-hand0-hand8":txt[33],
                        "Y-hand0-hand9":txt[34],
                        "Y-hand0-hand10":txt[35],
                        "Y-hand0-hand11":txt[36],
                        "Y-hand0-hand12":txt[37],
                        "Y-hand0-hand13":txt[38],
                        "Y-hand0-hand14":txt[39],
                        "Y-hand0-hand15":txt[40],
                        "Y-hand0-hand16":txt[41],
                        "Y-hand0-hand17":txt[42],
                        "Y-hand0-hand18":txt[43],
                        "Y-hand0-hand19":txt[44],
                        "Y-hand0-hand20":txt[45],
                        "Y-hand0-face6":txt[46],
                        "Y-hand0-face10":txt[47],
                        "Y-hand0-face27":txt[48],
                        "Y-hand0-face257":txt[49],},
                        index=["data"]
                    )
        df.to_csv(f"csv/distance_hand_face_{i}_{j}_FileName_label.csv")

"""
"Z-hand0-hand1":txt[50],
"Z-hand0-hand2":txt[51],
"Z-hand0-hand3":txt[52],
"Z-hand0-hand4":txt[53],
"Z-hand0-hand5":txt[54],
"Z-hand0-hand6":txt[55],
"Z-hand0-hand7":txt[56],
"Z-hand0-hand8":txt[57],
"Z-hand0-hand9":txt[58],
"Z-hand0-hand10":txt[59],
"Z-hand0-hand11":txt[60],
"Z-hand0-hand12":txt[61],
"Z-hand0-hand13":txt[62],
"Z-hand0-hand14":txt[63],
"Z-hand0-hand15":txt[64],
"Z-hand0-hand16":txt[65],
"Z-hand0-hand17":txt[66],
"Z-hand0-hand18":txt[67],
"Z-hand0-hand19":txt[68],
"Z-hand0-hand20":txt[69],
"Z-hand0-face6":txt[70],
"Z-hand0-face10":txt[71],
"Z-hand0-face27":txt[72],
"Z-hand0-face257":txt[73],
"""
```

# csvファイル結合
```merge_csv.py
import pandas as pd
import csv
import glob

sample_files = glob.glob('./test/*.csv')
list = []
for file in sample_files:
    print(file)
    list.append(pd.read_csv(file))

    df = pd.concat(list)
    df.to_csv('./test.csv',index=False)
```

# csvファイルを扱ったSVM
```svm_csv.py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score

y_true = []
for j in range(1, 5):
    for i in range(1, 61):
        y_true.append(j)

# csvファイルの読み込み
npArray = np.loadtxt("./train/trainD.csv", delimiter = ",", dtype = "float", skiprows = 1)

# 説明変数の格納
x = npArray[:, 2:]

# 目的変数の格納
y = npArray[:, 0:1].ravel()

# 学習手法にSVMを選択
model = svm.SVC(gamma=0.01, C=10., kernel='rbf')

# 学習
model.fit(x,y)

y_pred = []
for i in range(0,240):
    test = np.loadtxt("./test/testD.csv", delimiter = ",", dtype = "float")
    df = test[i:i+1]
    ans = model.predict(df)

    if ans == 1:
        print(i+1,int(ans),  "fuck")
        y_pred.append(int(ans))
    elif ans == 2:
        print(i+1,int(ans),  "good")
        y_pred.append(int(ans))
    elif ans == 3:
        print(i+1,int(ans),  "silent")
        y_pred.append(int(ans))
    elif ans == 4:
        print(i+1,int(ans),  "up")
        y_pred.append(int(ans))
    else:
        print(i+1,int(ans),  "不明")
        y_pred.append(int(ans))

tp = np.sum((np.array(y_true)==1)&(np.array(y_pred)==1))
tn = np.sum((np.array(y_true)==0)&(np.array(y_pred)==0))
fp = np.sum((np.array(y_true)==0)&(np.array(y_pred)==1))
fn = np.sum((np.array(y_true)==1)&(np.array(y_pred)==0))

accuracy_score = accuracy_score(y_true, y_pred)
error_rate = 1 - accuracy_score

print("正答率(予測がどれだけ正しいか)：", accuracy_score)
print("誤答率(予測がどれだけ誤っているか)：", error_rate)

print("tp:", tp)
print("tn:", tn)
print("fp:", fp)
print("fn:", fn)
```
