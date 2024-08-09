import os
import pandas as pd

def dir():
    path = r'Z:\ra\uhn\blad\Reports'
    return path

path = dir()

def reports():
    folder = path

    # List to store the names of subdirectories (subjects)
    subject_id = []
    report_id = []
    sub_path = []

        # Checking if the path exists
    if os.path.exists(path):
        # Getting list of items in the directory
        items = os.listdir(path)
        for item in items:
            # Checking if the item is a directory
            if os.path.isdir(os.path.join(path, item)):
                subject_id.append(item)

    # Sort subject_id by creation time (most recent first)
    #subject_id.sort(key=lambda x: os.path.getctime(os.path.join(path, x)), reverse=True)

    # Select the last folder
    # subject_id = subject_id[:5]

    for subject in subject_id:
        sub_path.append(os.path.join(folder, subject))

    for temp in sub_path:
        dir = os.listdir(temp)
        report_id.append([str(item) for item in dir])

    # Convert report id enteries as strings rather than lists
    report_id = [entry[0] for entry in report_id]

    #Create a dataframe from subject_id and report_id
    data = {'subject_id': subject_id, 'patientid': report_id}
    names = pd.DataFrame(data)

    # Excluding K029 from analysis
    names = names[names['subject_id'] != 'K029']

    # Creating a list of dictionaries with 'name' key
    TableID = [{'name': name} for name in report_id]

    # List of patient IDs
    patientid = [item['name'] for item in TableID]

    f = []
    files = []
    File_Name = []

    for i in range(len(report_id)):
        f.append(os.path.join(sub_path[i], report_id[i], 'reports'))  # Obtain the path of each 'reports' folder
        files.append([file for file in os.listdir(f[i]) if file.endswith('.csv')])  # Find all .csv files in 'reports' folder
        File_Name.append([file for file in files[i]])  # Extract the names of each .csv file

    # Convert patientid in 'names' to int
    names.iloc[:, 1] = names.iloc[:, 1].astype(int)

    # Creating a new column for notes
    notes = names.copy()
    notes['notes'] = None

    # Adding a note for VIDA case ID 2598
    var_1 = 'Automated segmentation did not work properly.'
    notes.loc[notes['patientid'] == 12, 'notes'] = var_1

    var_2 = 'Automated segmentation did not work properly.'
    notes.loc[notes['patientid'] == 61, 'notes'] = var_2

    var_3 = 'Automated segmentation did not work properly.'
    notes.loc[notes['patientid'] == 35, 'notes'] = var_3

    var_4 = 'Contains Contrast.'
    notes.loc[notes['patientid'] == 32, 'notes'] = var_4

    return patientid, f, File_Name, names, notes

