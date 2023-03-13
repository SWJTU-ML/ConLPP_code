from sklearn.preprocessing import StandardScaler
from src.metrics import knn_acc
from src.ConLPP import ConLPP  # our algorithm
import scipy.io as scio

if __name__ == '__main__':
    # load data'
    data = scio.loadmat('./data/indianliver.mat')
    x = data['data']
    y = data['labels'].flatten()

    ss_x = StandardScaler()
    x = ss_x.fit_transform(x)

    model = ConLPP(n_components=4) # target dimension = 4
    embedding = model.fit_transform(x)

    acc = knn_acc(embedding, y)
    print('the result is', acc * 100)
