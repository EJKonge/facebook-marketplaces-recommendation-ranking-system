import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class CategoryClassification:

    '''
    This Class will perform a SVC model to predict the category of product base on their picture and has the following functions:
    __init__(self)
    load_data(self)
    svc_model(self)
    main(self)
    '''
    def __init__(self) -> None:
        #self.path = 'path to folder'
        self.test_to_train_ratio = 0.3
        self.seed = 37

    def load_data(self):

        '''
        Loads raw data, remove null values and stop words and, defines the lables and target
        RETURNS
        -------
        X, y
            It returens labels (X) and the target (y)
        '''
        df = pd.read_pickle(self.path + 'products+arrays.pkl')
        shuffled = df.sample(frac=1, random_state=37).reset_index()

        X = shuffled['image']
        y = shuffled['category_id']
        label = pd.DataFrame(y)
        X_tmp = np.asarray([np.asarray(shuffled['image'][x]) for x in range(shuffled.shape[0])])
        X_tmp=np.reshape(X_tmp,(shuffled.shape[0], 1, 64, 64))
        nsamples, nx, ny, nz = X_tmp.shape

        X = X_tmp.reshape((nsamples,nx*ny*nz))
        y = label.apply(lambda x: x.argmax(), axis=1).values
        return X, y

    def svc_model(self):

        '''
        Scales the image arrays and, builds a SVC model to predic the category of the products
        '''
        X, y = self.load_data()

        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size = self.test_to_train_ratio, random_state = self.seed)
        
        clf = SVC(kernel='linear', probability=True, random_state=37)
        clf.fit(list(X_train), y_train)

        y_pred = clf.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_pred, y_test)
        print('Model accuracy is: ', accuracy)
        print('Score: ', clf.score(X_test, y_test))

    def main(self):
        self.svc_model()