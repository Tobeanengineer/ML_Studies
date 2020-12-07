import numpy as np

'''
   train : 전체 데이터 / test : 입력 데이터 / label : 데이터의 라벨이 담겨있는 리스트
   8~15th : 입력 데이터와 전체 데이터 간 유클리드 거리를 구한 후 distance_list에 추가, 데이터의 라벨도 같이 추가해줌
   17th : 거리를 기준으로 정렬
   19~22nd : 입력 데이터를 기준으로 주변 k개의 데이터 라벨과 갯수를 딕셔너리에 저장
   26th : items 함수를 통해 뽑은 [(라벨, 개수)] 데이터를 개수 기준(-x[1])으로 내림차순 정렬
   29th : 입력 데이터 주변으로 가장 많이 있는 데이터가 result의 맨 앞에 정렬되있으므로 그 라벨을 return
'''

def KNN(k, train, test, label):
    train, test = np.array(train), np.array(test)
    size = len(train)
    distance_list = []
    class_label = {}

    for i in range(size):
        distance = np.sqrt(sum((train[i]-test)**2))
        distance_list.append([distance, label[i]])

    distance_list.sort(key=lambda x: x[0])

    for j in distance_list[:k]:
        class_label[j[1]] = 1
        if class_label[j[1]]:
            class_label[j[1]] += 1

    result = sorted(class_label.items(), key=lambda x: -x[1])

    return result[0][0]

k = 5
train = [[1,2], [2, 4], [3, 4], [4, 5], [5, 6], [7, 4], [1, 3], [9, 7]]
test = [8, 7]
label = ['A','A','A','A','B','B','B','B']
print(KNN(k, train, test, label))
