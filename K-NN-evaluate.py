import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import io, transform

# 画像をロードして前処理する関数
def load_and_preprocess_image(filepath, target_size=(64, 64)):
    image = io.imread(filepath)
    image_resized = transform.resize(image, target_size)
    return image_resized.flatten()

# サンプルデータ（猫と犬の画像ファイルパスをそれぞれ100個ずつ準備する）
# cat_images = ['path/to/cat/image1.jpg', 'path/to/cat/image2.jpg', ..., 'path/to/cat/image100.jpg']
# dog_images = ['path/to/dog/image1.jpg', 'path/to/dog/image2.jpg', ..., 'path/to/dog/image100.jpg']

# データセットの作成
cat_images = [f'.\\PetImages\\Cat\\{i}.jpg' for i in range(0, 100)]
dog_images = [f'.\\PetImages\\Dog\\{i}.jpg' for i in range(0, 100)]

X = []
y = []

for image_path in cat_images:
    X.append(load_and_preprocess_image(image_path))
    y.append(0)  # 猫のラベル

for image_path in dog_images:
    X.append(load_and_preprocess_image(image_path))
    y.append(1)  # 犬のラベル

X = np.array(X)
y = np.array(y)

# K-NNモデルのトレーニング
k = 100
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# 新しい画像を分類
new_image_path = '.\\PetImages\\Cat\\50.jpg'
new_image = load_and_preprocess_image(new_image_path)

# 近傍点の取得
distances, indices = knn.kneighbors([new_image], n_neighbors=k)

# 近傍点のラベルを取得
neighbor_labels = y[indices[0]]

# 多数決による分類
predicted_label = np.bincount(neighbor_labels).argmax()

# 正しいラベルの数を出力
correct_label_count = np.sum(neighbor_labels == predicted_label)

print(f'Predicted Label: {"Dog" if predicted_label == 1 else "Cat"}')
print(f'Correct Label Count: {correct_label_count}')
