from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

class RandomForest(FlowSpec):
    """
    RandomForest is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
  
    DATA_FILE = IncludeFile(
        'dataset',
        help='csv file with the dataset',
        is_text=False,
        default='loan_dataset.csv')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )

    VALIDATION_SPLIT = Parameter(
        name='validation_split',
        help='validation dataset size',
        default=0.20
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        import pandas as pd
        self.df = pd.read_csv('loan_dataset.csv')
        self.df = self.df.dropna()
        cols = [ 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 
        'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']
        self.Xs = self.df[cols]
        self.Ys = LabelEncoder().fit_transform(self.df['Loan_Status'])
        # go to the next step
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        from sklearn.model_selection import train_test_split    
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.compose import make_column_selector as selector

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
        )

        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, selector(dtype_exclude="object")),
                ("cat", categorical_transformer, selector(dtype_include="object")),
            ]
        )
        self.Xs = preprocessor.fit_transform(self.Xs)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xs, 
            self.Ys, 
            test_size=self.TEST_SPLIT, 
            random_state=42
            )
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train a regression on the training set
        """
        from sklearn.ensemble import RandomForestClassifier

        reg = RandomForestClassifier()
        reg.fit(self.X_train, self.y_train)
        # now, make sure the model is available downstream
        self.model = reg
        # go to the testing phase
        self.next(self.test_model)

    @step 
    def test_model(self):
        """
        Test the model on the hold out sample
        """
        from sklearn import metrics
        import numpy as np
        pred = self.model.predict(self.X_test)
        self.acc = metrics.accuracy_score(self.y_test, pred)
        print('The  model based on Accuracy has Accuracy : {}'.format(self.acc))        
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    RandomForest()