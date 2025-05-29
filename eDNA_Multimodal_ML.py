# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pycaret.classification import ClassificationExperiment, load_model, predict_model
import shap
import matplotlib.pyplot as plt

###### DATA PREPROCESSING #####
#---- eDNA sequence processing ----#
# Load non-chimeric ASV sequences
eDNA_seq = pd.read_csv('Data/merged_seqtab_nochim.csv')
eDNA_seq = pd.melt(eDNA_seq, id_vars='DNA_SEQ', var_name='sample', value_name='read_count')

# Keep sequences > 10 reads
eDNA_seq['seq_length'] = [len(seq) for seq in eDNA_seq.DNA_SEQ]
eDNA_seq = eDNA_seq[eDNA_seq['read_count']>10]

# Keep sequences between 136 bp and 148 bp
eDNA_seq = eDNA_seq.loc[(eDNA_seq['seq_length']>136) & (eDNA_seq['seq_length']<148)]

# Remove ASVs found in > 0.1% in negative control
neg_control = pd.read_csv('Data/samples_meta.csv')
neg_control = neg_control[neg_control['type']=='n_control']
neg_control = neg_control.merge(eDNA_seq, how='left', on='sample')
neg_control = neg_control.groupby(['DNA_SEQ'])['read_count'].sum().reset_index()
ASV_count_sum = eDNA_seq.groupby(['DNA_SEQ'])['read_count'].sum().reset_index()
neg_control = neg_control.merge(ASV_count_sum, how='left', on='DNA_SEQ')
neg_control['ratio'] = neg_control['read_count_x'] / neg_control['read_count_y']
neg_control = neg_control[neg_control['ratio']>0.01]
eDNA_seq = eDNA_seq[~eDNA_seq['DNA_SEQ'].isin(neg_control['DNA_SEQ'])]

# Normalise reads to 0 - 1 range by sample
# Function to normalise reads using MinMax scaler
def scale_read(x):
    """
    Normalize read counts in row x and return results in 'read_count_scaled' column.
    :param x: row in dataframe
    :return: normalized read counts in 'read_count_scaled' column.
    """
    scaler = MinMaxScaler()
    x['read_count_scaled'] = scaler.fit_transform(x['read_count'].values.reshape(-1, 1))
    return x

# Normalise ASV read counts
eDNA_seq = eDNA_seq.groupby('sample', group_keys=False).apply(lambda x: scale_read(x))

#---- Metadata processing ----#
# Load metadata for sample, land and site
site_meta = pd.read_csv('Data/sites_meta.csv')
sample_meta = pd.read_csv('Data/samples_meta.csv')
sample_meta = sample_meta.dropna()  # remove controls
land_meta = pd.read_csv('Data/landuse_3cat.csv')

# Filter data to match eDNA/ASV dataset
sample_meta = sample_meta[sample_meta['sample'].isin(eDNA_seq['sample'].unique())]
site_meta = site_meta[site_meta['site_code'].isin(sample_meta['site_code'].unique())]
site_meta = site_meta[['site_code', 'Biogeo', 'Altitude', 'Catchment']] # select relevant biogeographic data

# Assign site status as reference (forest proportion =>33%) or impacted
land_meta = land_meta[land_meta['description']=='Forest']
land_meta['status'] = np.where(land_meta['lu_prop']>=0.33, 'Ref', 'Imp')

# Merge metadata for sample, site and land
metadata = sample_meta.merge(site_meta).merge(land_meta)

# Create 'site_season' column
metadata['site_season'] = metadata['site_code'] + '-' + metadata['season']

# Merge eDNA with metadata
eDNA_seq = eDNA_seq.merge(metadata)
eDNA_seq = eDNA_seq.dropna()

#---- Train-Test Split ----#
# Create dataset using 'site_season' as sample (x) and site status as target (y)
dataset = eDNA_seq.groupby(['site_season', 'status'])['season'].nunique().reset_index()
x = dataset['site_season']
y = dataset['status']

# Train-test split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    stratify=y,
                                                    test_size=test_size,
                                                    random_state=52)
# Split eDNA data according to train-test split
x_train = eDNA_seq[eDNA_seq['site_season'].isin(X_train)]
x_test = eDNA_seq[eDNA_seq['site_season'].isin(X_test)]

# Assign id to ASVs in train set (used in DA analysis)
ASV_id = x_train.groupby('DNA_SEQ')['read_count'].sum().reset_index()
ASV_id = ASV_id.drop(columns=['read_count'])
ASV_id['ID'] = [f'ASV{n+1}' for n in range(ASV_id.shape[0])]

##### MULTIMODAL ML WORKFLOW #####
#---- STEP 1: Differentially Abundant ASVs ----#
# Import DA analysis results (example given for differential ranking, DR)
df_DR = pd.read_csv('Data/DR_differentials.tsv', sep="\t")
df_DR = df_DR.rename(columns={'featureid':'ID'})
df_DR = df_DR.merge(ASV_id)

# # For ancombc2:
# df_ancombc2 = pd.read_csv('Data/ancombc2_output.csv', sep=",")
# df_ancombc2 = df_ancombc2.rename(columns={'taxon':'ID'})
# df_ancombc2 = df_ancombc2.merge(ASV_id)

# Assign taxonomic labels to ASVs
df_taxa = pd.read_csv('Data/merged_taxo_IdTaxa.csv')
df_DR = df_DR.merge(df_taxa)

#---- STEP 2: Feature Filtering based on LFC ----#
df_DR['abs_LFC'] = abs(df_DR['Ref_or_Imp[T.Ref]'])  # get absolute LFC values
percentile_LFC = 40
threshold = np.percentile(df_DR['abs_LFC'], percentile_LFC)
ASV_filtered = df_DR[df_DR['abs_LFC']>threshold]  # resulting ASV sequences serve as absolute labels/reference

# Construct feature table using ASV sequences in ASV_filtered
ASV_ft = eDNA_seq[eDNA_seq['DNA_SEQ'].isin(ASV_filtered['DNA_SEQ'])]
ASV_ft = ASV_ft.merge(ASV_filtered)
ASV_ft = ASV_ft.groupby(['site_season', 'ID'])['read_count'].sum().reset_index()
ASV_ft = ASV_ft.groupby(['site_season'], group_keys=False).apply(lambda x: scale_read(x))  # normalise reads by sample
ASV_ft = ASV_ft.pivot(index='site_season', columns='ID', values='read_count_scaled').reset_index()
ASV_ft = ASV_ft.fillna(0)

#---- STEP 3: Feature Merging ----#
biogeo_data = metadata[['site_season','status', 'Altitude', 'Biogeo', 'Catchment']].drop_duplicates()
multimodal_data = ASV_ft.merge(biogeo_data)

# Multimodal data train-test split
train_DR_multimodal = multimodal_data[multimodal_data['site_season'].isin(X_train)].drop(columns=['site_season'])
test_DR_multimodal = multimodal_data[multimodal_data['site_season'].isin(X_test)].drop(columns=['site_season'])

# eDNA only data train-test split
train_DR = train_DR_multimodal.drop(columns=['Altitude', 'Biogeo', 'Catchment'], axis=1)
test_DR = test_DR_multimodal.drop(columns=['Altitude', 'Biogeo', 'Catchment'], axis=1)

#---- STEP 4: Automated Machine Learning for Impact Prediction ---#
session_id = 123
s = ClassificationExperiment()
s.setup(train_DR_multimodal,  # replace with train_DR for ML using eDNA only
        target='status',
        session_id=session_id,
        use_gpu=True,  # enable GPU for faster training
        test_data=test_DR_multimodal,  # replace with test_DR for ML using eDNA only
        index=False)

#List of ML models
model_list = ['catboost',
              'xgboost',
              'lightgbm',
              'rbfsvm',
              'mlp',
              'rf']

# Train ML models and evaluate on test set (classification report)
for model_name in model_list:
    model = s.create_model(model_name)
    model_tuned = s.tune_model(model,  # tune hyperparameters, this will take some time (!)
                               search_library='optuna',  # Optuna tuner
                               n_iter=100,
                               choose_better=True)
    model_prediction = s.predict_model(model_tuned)
    y_true = s.y_test
    y_predict = model_prediction['prediction_label']
    cls_report = classification_report(y_true, y_predict, digits=4)  # generate classification report
    with open(f'Models/{model_name}.txt', 'w') as file:  # save classification report to Models folder
        file.write(f'#### Classification report for {model_name} on test set ####\n'
                       f'{cls_report}')
    s.save_model(model_tuned, f'Models/{model_name}')  # save model pipeline to Models folder

#---- STEP 5: Model Explanation Using SHAP values ----#
# This step is demonstrated using the best multimodal MLP model reported in the study.
# Load MLP PyCaret pipeline
mlp_pipeline = load_model('Models/multimodal_mlp')

# Evaluate MLP pipeline on test set
test_x = test_DR_multimodal.drop(['status'], axis=1)
y_predict = predict_model(mlp_pipeline, test_x)['prediction_label']
y_true = test_DR_multimodal['status']
cls_report = classification_report(y_true, y_predict, digits=4)
print(cls_report)  # accuracy should be 0.8333

# Retrieve MLP model
model = mlp_pipeline.named_steps['trained_model']

# Preprocess train set using pipeline
train_x = train_DR_multimodal.drop(['status'], axis=1)
train_x = mlp_pipeline.transform(train_x)

# Calculate SHAP values using Kernel Explainer
explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(train_x, 10))
shap_values = explainer.shap_values(train_x)

# SHAP bar plot for class 1 (ref) prediction
plt.figure(dpi=600)
shap.summary_plot(shap_values[1], train_x, plot_type='bar',
                  max_display=10, show=False)  # display top 10 features
plt.savefig('SHAP_barplot.jpg',bbox_inches='tight')  # save figure

# SHAP beeswarm plot for class 1 (ref) prediction
plt.figure(dpi=600)
shap.summary_plot(shap_values[1], train_x, plot_type='dot',
                  max_display=10, show=False)  # display top 10 features
plt.savefig('SHAP_beeswarm.jpg',bbox_inches='tight')  # save figure
