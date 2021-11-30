ALL_DATASETS = ['cornell-sent-polarity', 'cornell-sent-subjectivity', 'ag_news2', 'ag_news3', 'dbpedia3', 'dbpedia8']
DATASETS = ['cornell-sent-polarity', 'ag_news3', 'dbpedia3']
#BIG_DATASETS = ['ag_news2_big', 'ag_news3_big', 'dbpedia3_big', 'dbpedia8_big']
HUE_ORDER = ['strong', 'medium', 'weak']


def reg_map(x):
    if x.endswith('L1'):
        return 'L1(C=1)'
    if not x[-1].isdigit():
        c = 1
    else:
        first_digit = min([i for i in range(len(x)) if x[i].isdigit()])
        c = float(x[first_digit:].replace(',', '.'))
        c = 1/c
    #if x.startswith('RegressionStable'):
    #    c = c/1000
    if int(c) == c:
        return str(int(c))
    return str(round(c,5)).replace('.',',')

def regularization_to_string(model_name, norm_type):
    mapping = {'0,1': 'weak', '1': 'medium', '10': 'strong'}
    if model_name.startswith('Regression'):
        mapping = {'0,01': 'weak', '0,1': 'medium', '1': 'strong'}
        
    c = reg_map(model_name)
    if norm_type == 'l1':
        mapping = {'0,001': 'weak', '0,01': 'medium', '0,1': 'strong'}
        if model_name.startswith('svmHinge'):
            mapping = {'0,01': 'weak', '0,1': 'medium', '1': 'strong'}
    return mapping.get(c, 'unknown')
    

def model_type(x):
    if x.startswith('Regression'):
        if x.startswith('RegressionStable'):
            return 'RegressionStable'
        return 'Regression'
    if x.startswith('svmLinear'):
        return 'svm-Hinge-Squared'
    return 'svm-Hinge'