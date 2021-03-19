import datarobot as dr
import pandas as pd
from functools import reduce
import umap
import hdbscan
import numpy as np

class PredictionExplanationsClustering:
    """
    Attributes:
        project_id    <string> The project id of your completed DataRobot project.
        model_id      <string> The model id of the model you want to use to produce prediction explanations (pe).
        data          <pandas dataframe> or <path_to_file> The data that will be used to upload and scored for pe.
        n_reasons     <int> number of reasons to produce. Can take a maximum value of 10.
    """
    
    def __init__(self, project_id, model_id, data, n_reasons = 5):
        self.project_id = project_id
        self.model_id = model_id
        self.data = data
        self.project = dr.Project.get(self.project_id)
        self.model = dr.Model(id=model_id, project_id = project_id)
        self.n_reasons = n_reasons
        self.prediction_explanation_results = None
        self.strength_per_feature_cols = None
        self.clusterable_embedding = None
        self.labels = None
        self.results = None
        
    def retrieve_prediction_explanations(self):
        """This function will retrieve prediction explanations for the 
        dataset you use when defining the PredictionExplanationsClustering object.
        
        To see the results of this function, call 'PredictionExplanationsClustering.prediction_explanation_results'.
        """
        
        print('Initiating calculation of prediction explanations. \nThis may take a while to complete...')
        
        # Upload Dataset and get prediction results
        dataset = self.project.upload_dataset(self.data) # Returns an instance of [PredictionDataset]
        pred_job = self.model.request_predictions(dataset.id)
        preds = pred_job.get_result_when_complete()
        
        # Ensure Feature Impact exists
        try:
            impact_job = self.model.request_feature_impact()
            impact_job.wait_for_completion()
        except dr.errors.JobAlreadyRequested:
            pass  # already computed
        
        # Ensure Prediction Explanations are computed
        try:
            dr.PredictionExplanationsInitialization.get(self.project.id, self.model.id)
        except dr.errors.ClientError as e:
            assert e.status_code == 404  # haven't been computed
            init_job = dr.PredictionExplanationsInitialization.create(self.project.id, self.model.id)
            init_job.wait_for_completion()
            
        # Run the reason code job
        rc_job = dr.PredictionExplanations.create(self.project.id,
                                   self.model.id,
                                   dataset.id,
                                   max_explanations=self.n_reasons,
                                   threshold_low=None,
                                   threshold_high=None)
        
        rc = rc_job.get_result_when_complete(max_wait = 10000)
        all_rows = rc.get_all_as_dataframe()
        self.prediction_explanation_results =  all_rows
        
        
    # Some utility functions used by our methods.
    def __unlist(self, listOfLists):
        return [item for sublist in listOfLists for item in sublist]
    
    def __unique_elements(self, bigList):
        return reduce(lambda l, x: l.append(x) or l if x not in l else l, bigList, [])
    
    def create_column_name(self, cluster_index, suffix=''):
        cluster_index = int(cluster_index) # we expect cluster indexes to be integers
        if cluster_index == -1:
            column_name = 'Noise'
        else:
            column_name = 'cluster_{}'.format(str(cluster_index).upper())
        column_name = column_name + suffix
        return column_name

    def get_strength_per_feature_cols(self):
        """This function preprocesses the results of retrieve_prediction_explanations 
        method to produce a dataframe of strength per feature.
        
        To see the results of this function, call 'PredictionExplanationsClustering.strength_per_feature_cols'.
        """
        
        print('Preprocessing results to produce a dataframe of strength per feature...')
        
        colsToUse = []
        startPoint = 6
        if self.project.target_type == 'Regression':
            startPoint = 2
        if self.project.target_type == 'Binary':
            startPoint = 6
        j = startPoint
        for i in range(self.n_reasons):
            colsToUse.append(j) 
            colsToUse.append(j+4)
            j = j + 5 
        try:
            rc3 = self.prediction_explanation_results.iloc[:, colsToUse]
        except:
            raise Exception("Please execute 'retrieve_prediction_explanations' method first.")
        j = 0
        colsForNames = []
        for i in range(self.n_reasons):
            colsForNames.append(j) 
            j = j + 2 
        namesdf = rc3.iloc[:,colsForNames]
        allfeatures = [namesdf[i].unique().tolist() for i in namesdf.columns]
        nameslist = self.__unique_elements(self.__unlist(allfeatures))
        
        # Create a new dataframe with one column per possible reason code. 
        # Initialise with zero and fill in with the explanation strenghts.
        dfnew = pd.DataFrame(columns=nameslist)
        for j in range(len(rc3)):
            dfnew.loc[j] = [0 for n in range(len(nameslist))]
            for i in range(self.n_reasons):
                rcname = rc3.loc[j][i*2] 
                rcvalue = rc3.loc[j][i*2+1]
                dfnew.loc[j][rcname] = rcvalue    
                
        self.strength_per_feature_cols = dfnew
        
    def run_umap(self, n_neighbors=30, min_dist=0.0, n_components=2):
        
        """This function applies umap dimentionality reduction on the 'strength_per_feature_cols' dataset.
        """
        
        print('Applying UMAP dimentionality reduction...')
        
        try:
            self.clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, 
                                              min_dist=min_dist,
                                              n_components = n_components,
                                              random_state = 42).fit_transform(self.strength_per_feature_cols)
        except:
            raise Exception("Please execute 'retrieve_prediction_explanations' and 'get_strength_per_feature_cols' methods first.")
            
    def run_hdbscan(self,min_samples=10, min_cluster_size=500):
        """This function applies hdbscan clustering on the data after umap has been applied."
        """
        print('Applying HDBSCAN clustering...')
        
        try:
            self.labels = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size).fit_predict(self.clusterable_embedding)
        except:
            raise Exception("Please execute 'retrieve_prediction_explanations', 'get_strength_per_feature_cols'  and 'run_umap' methods first.")
            
    def get_results(self):
        """This function summarizes the prediction strength actual feature value by cluster."""
        print('Calculating results...')

        # Summarize prediction exstrengths by cluster

        self.strength_per_feature_cols['clusters'] = self.labels
        self.strength_per_feature_cols['predicted_score'] = self.prediction_explanation_results['prediction']

        self.strength_per_feature_cols = self.strength_per_feature_cols.astype(np.float64)
        expl_str_by_cluster = pd.DataFrame(self.strength_per_feature_cols.groupby('clusters').mean().T)
        expl_str_by_cluster = expl_str_by_cluster.rename(columns = lambda x: self.create_column_name(x, '_expl_mean'))

        # Summarize actual feature values by cluster
        self.data['clusters']=self.labels
        self.data['predicted'] = self.prediction_explanation_results['prediction']

        expl_actual_by_cluster = pd.DataFrame(self.data.groupby('clusters').mean().T)
        expl_actual_by_cluster = expl_actual_by_cluster.rename(columns=lambda x: self.create_column_name(x, '_actual_mean'))

        #Save final result after joining the two dataframes
        final_result = expl_actual_by_cluster.join(expl_str_by_cluster, how='left')
        self.results = final_result
        
    def run_full_procedure(self):
        self.retrieve_prediction_explanations()
        self.get_strength_per_feature_cols()
        self.run_umap()
        self.run_hdbscan()
        self.get_results()
