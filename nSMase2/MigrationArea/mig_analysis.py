import datetime
import os
import glob
import numpy as np
import pandas as pd

def get_analysis_date():
    
    now = datetime.datetime.now()
    analysis_date = now.strftime("%Y%m%d")
    
    return analysis_date

def read_imagej_csv(path):
    import os
    import glob
    import numpy as np
    import pandas as pd
    # Test all the files meet requirements
    check_files_in_path_folder(path)

    # Get list of files in folder
    files = glob.glob(path + "/*.csv")
    num_files = len(files)
    # Set up empty lists to store values for dataframe
    control_area = np.empty(num_files)
    expt_area = np.empty(num_files)
    image_names = []
    dates = []
    treatments = []
    somite_num = []
    doses = []
    
    for i in range(len(files)):
        # Read the csv into a dataframe
        df = pd.read_csv(files[i])
        

        # Get image name and add it to image name list
        image = os.path.splitext(os.path.basename(files[i]))[0]
        image_names.append(image)
        
        
        # Each csv represents 1 embryo. Get control and experimental migration area for each embryo.
        control_area[i] = df['Area'][0]
        expt_area[i] = df['Area'][1]
        

        # Get experiment information from image name and add it to appropriate lists
        date, treatment, dose, stains, embryo, somites, imagemag = image.split('_')
        dates.append(date)
        treatments.append(treatment)
        somite_num.append(somites)
        doses.append(dose)
        
    # Create dictionary of data to use in dataframe    
    data_dict = {'Date' : dates, 'Treatment' : treatments, 'Dose' : doses, 'Image' : image_names, 'Somites' : somite_num, 
                 'Control Area': control_area, 'Experiment Area' : expt_area}

    # Create dataframe
    df_data = pd.DataFrame(data_dict)
    assert len(df_data) == num_files, 'Length of dataframe different than number of files!'
    return df_data

def calculate_norm_values(df):
    df['Exp/Ctl Area'] = df['Experiment Area']/df['Control Area']
    
    # Create empty arrays to store normalized control and experimental areas
    norm_cnt_area = np.empty(len(df))
    norm_exp_area = np.empty(len(df))  

    # Get list of unique treatments
    treatment_list = df.Treatment.unique()

    
    for treat in treatment_list:
        # For each treatment, get indexes of the embryos for that treatment
        treat_rows = df.loc[df['Treatment'] == treat]
    
        # Get mean control and experiment area for that treatment
        cntl_mean = treat_rows['Control Area'].mean()
        expt_mean = treat_rows['Experiment Area'].mean()
        inds = treat_rows.index.values
        
        # Calculate normalized areas
        for i in inds:
            norm_cnt_area[i] = df['Control Area'][i] / cntl_mean
            norm_exp_area[i] = df['Experiment Area'][i] / cntl_mean

    # Create columns for normalized areas
    df['Norm Control Area'] = norm_cnt_area
    df['Norm Experiment Area'] = norm_exp_area
    
    return df

def get_ultimate_dataframe(path):

	df = read_imagej_csv(path)
	df_data = calculate_norm_values(df)

	return df_data

def print_stats(df):
    df_mean = df.groupby('Treatment')['Control Area', 'Experiment Area', 'Exp/Ctl Area'].mean()
    df_sem = df.groupby('Treatment')['Control Area', 'Experiment Area', 'Exp/Ctl Area'].sem()
    df_corr = df.groupby('Treatment')['Control Area', 'Experiment Area'].corr()

    print('Mean')
    print(df_mean)
    print('\n')
    
    print('SEM')
    print(df_sem)
    print('\n')
    
    print('Correlation')
    print(df_corr)



def get_stats_df(df):
    treatment_list = df.Treatment.unique()
    cont_mean = []
    cont_sem = []
    expt_mean = []
    expt_sem = []
    expt_div_cont_mean = []
    expt_div_cont_sem = []
    corr = []
    
    for treatment in treatment_list:
        inds = df.loc[df['Treatment'] == treatment]
        cont_mean.append(inds['Control Area'].mean())
        cont_sem.append(inds['Control Area'].sem())
        expt_mean.append(inds['Experiment Area'].mean())
        expt_sem.append(inds['Experiment Area'].sem())
        expt_div_cont_mean.append(inds['Exp/Ctl Area'].mean())
        expt_div_cont_sem.append(inds['Exp/Ctl Area'].sem())
        correlations = inds.corr()
        corr.append(correlations['Experiment Area'][0])
    
    stats_dict = {'Treatment' : treatment_list, 'Control Area Mean' : cont_mean, 'Control Area SEM' : cont_sem, 
                 'Experiment Area Mean' : expt_mean, 'Experiment Area SEM' : expt_sem, 
                 'Exp/Ctl Mean' : expt_div_cont_mean, 'Exp/Ctl SEM' : expt_div_cont_sem, 
                 'Correlation' : corr}
    df_stats = pd.DataFrame(stats_dict)
    return df_stats    


def check_files_in_path_folder(path):
    import glob
    files = glob.glob(path + "/*.csv")
    num_files = len(files)
    print(f'Number of files in this folder: {num_files}')
    print('\n')
    
    df_csv1 = pd.read_csv(files[0])
    column_names = df_csv1.columns.values
    csv1_length = len(df_csv1)
    
    print('In first file:')
    print(f'Number of rows: {csv1_length}')
    print(f'Number of columns: {len(column_names)}')
    print(f'Column names: {column_names}')
    
    image1 = os.path.splitext(os.path.basename(files[0]))[0]
    print(f'Image name: {image1}')
    
    image_components = image1.split('_')
    print('Image name components:')
    for c in image_components:
        print(c)
    
    print('\n')
    error_files = []
    for i in range(num_files):
        # Read the csv into a dataframe
        df = pd.read_csv(files[i])
        filename = os.path.basename(files[i])
        
        # Get image name
        image = os.path.splitext(os.path.basename(files[i]))[0]

        if len(df) != csv1_length:
            error_files.append(filename)
            print(f'ERROR! Expected {len(df_csv1)} rows in csv, file {filename} has {len(df)} rows!')
        
        if df.columns.values.all() != column_names.all():
            error_files.append(filename)
            print(f'ERROR! Expected columns: {column_names} \n {filename} has columns: {df.colunms.values} ')
        
        if len(image.split('_')) != len(image_components):
            error_files.append(filename)
            print(f'ERROR! Expected {len(image_components)} components in image name. \n')
            print(f'Check this file\'s image name: {filename}')
        
    assert len(error_files) == 0, f'These files do not match expected format: {error_files}. Check docstring for csv requirements.'
    if len(error_files) == 0:
        print('You\'re good to go! All files match expected format.')



