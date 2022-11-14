from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# MAKE SURE THESE VARIABLES HAVE BEEN SET
os.environ['COMET_API_KEY'] = 'CZ1Ta6S1B7OSJpgNaXr4zBXbg'
os.environ['MY_PROJECT_NAME'] = 'fz2101-hw5'    

assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'

from comet_ml import Experiment


class RandomForest(FlowSpec):
    """
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
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
    Max_depth = Parameter('max_depth',
                    help="maximum depth of random forest.",
                    default="1,3,5,10,20,50",
                    separator=',')
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
        self.x_train,self.x_vali,self.y_train,self.y_vali=train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=self.VALIDATION_SPLIT / (1 - self.TEST_SPLIT), 
            random_state=42
            )
        self.next(self.train_model,foreach="Max_depth")

    @step
    def train_model(self):
        """
        Train a regression on the training set
        """
        from sklearn.ensemble import RandomForestClassifier

        self.max_depth=int(self.input)
        reg = RandomForestClassifier(self.max_depth)
        reg.fit(self.x_train, self.y_train)
        # now, make sure the model is available downstream
        self.model = reg
        # go to the testing phase
        self.next(self.validation)



    @step 
    def validation(self,inputs):
        """
        Test the model on the hold out sample
        """
        from sklearn import metrics
        import numpy as np
        #self.exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'],
                    #auto_param_logging=False)
        self.accs=[]
        self.f_1s=[]
        self.models=[input.model for input in inputs]
        self.max_depths=[input.max_depth for input in inputs]
        self.merge_artifacts(inputs, include=['x_train','y_train','x_vali','y_vali','X_test','y_test'])
        for input in inputs:
            self.y_predicted = input.model.predict(input.x_vali)
            self.acc = metrics.accuracy_score(input.y_vali, self.y_predicted)
            self.f1 = metrics.f1_score(input.y_vali, self.y_predicted)
            self.accs.append(self.acc)
            self.f_1s.append(self.f1)
        

        for i in range(len(self.accs)):
            self.mse=self.accs[i]
            self.r2=self.f_1s[i]
            self.max_depth=self.max_depths[i]
            self.model=self.models[i]

        self.idx = np.argmax(self.accs)
        self.acc=self.accs[self.idx]
        self.f1=self.f_1s[self.idx]
        self.max_depth=self.max_depths[self.idx]
        self.model=self.models[self.idx]
        print('The best model based on Accurcy has Accuracy : {}, f1 score : {}'.format(self.acc,self.f1))

        all_validation_statistics = {"max_depth_1": self.max_depths[0],
                                        "max_depth_2": self.max_depths[1],
                                        "max_depth_3": self.max_depths[2],
                                        "max_depth_4": self.max_depths[3],
                                        "max_depth_5": self.max_depths[4],
                                        "max_depth_6": self.max_depths[5],
                                        "acc_1": self.accs[0],
                                        "acc_2": self.accs[1],
                                        "acc_3": self.accs[2],
                                        "acc_4": self.accs[3],
                                        "acc_5": self.accs[4],
                                        "acc_6": self.accs[5],
                                        "f1_1": self.f_1s[0],
                                        "f1_2": self.f_1s[1],
                                        "f1_3": self.f_1s[2],
                                        "f1_4": self.f_1s[3], 
                                        "f1_5": self.f_1s[4],
                                        "f1_6": self.f_1s[5],                                                                           
                                      "Best_max_depth": self.max_depth,
                                      "Best_f1":self.f1,
                                      "Best_acc":self.acc}
        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'], auto_param_logging=False)
        exp.log_dataset_hash(self.x_train)
        
        
        exp.log_parameters({"Max_depth": self.max_depth})
        exp.log_metrics(all_validation_statistics)
        
        self.next(self.end)

    

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    RandomForest()