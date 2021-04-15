import numpy as np
def convert_data(data, dtype, data_order, data_format):
    '''
    Convert input data (numpy array) to needed format, type and order
    '''
    # Firstly, change order and type of data
    if data_order == 'F':
        data = np.asfortranarray(data, dtype)
    elif data_order == 'C':
        data = np.ascontiguousarray(data, dtype)

    # Secondly, change format of data
    if data_format == 'numpy':
        return data
    elif data_format == 'pandas':
        import pandas as pd

        if data.ndim == 1:
            return pd.Series(data)
        else:
            return pd.DataFrame(data)
    elif data_format == 'cudf':
        import cudf
        import pandas as pd

        return cudf.DataFrame.from_pandas(pd.DataFrame(data))

def load_data(shape=(1000000, 50), generated_data=[], add_dtype=False, label_2d=False,
              int_label=False):
    full_data = {
        file: None for file in ['X_train', 'X_test', 'y_train', 'y_test']
    }
    #param_vars = vars(params)
    #print(full_data)
    int_dtype = np.int32 if '32' in 'numpy.float64' else np.int64
    for element in full_data:
        # generate and convert data if it's marked and path isn't specified
        if full_data[element] is None and element in generated_data:
            full_data[element] = convert_data(
                np.random.rand(*shape),
                int_dtype if 'y' in element and int_label else np.float64,
                'C', 'numpy')
        # convert existing labels from 1- to 2-dimensional
        # if it's forced and possible
        if full_data[element] is not None and 'y' in element and label_2d and hasattr(full_data[element], 'reshape'):
            full_data[element] = full_data[element].reshape(
                (full_data[element].shape[0], 1))
        # add dtype property to data if it's needed and doesn't exist
        if full_data[element] is not None and add_dtype and not hasattr(full_data[element], 'dtype'):
            if hasattr(full_data[element], 'values'):
                full_data[element].dtype = full_data[element].values.dtype
            elif hasattr(full_data[element], 'dtypes'):
                full_data[element].dtype = full_data[element].dtypes[0].type

    # clone train data to test if test data is None
    for data in ['X', 'y']:
        if full_data[f'{data}_train'] is not None and full_data[f'{data}_test'] is None:
            full_data[f'{data}_test'] = full_data[f'{data}_train']
    return tuple(full_data.values())

import numpy as np
from sklearn.linear_model import LinearRegression
import time

# generate regression dataset
print()
row = 2000000
col = 100
print(f'Starting generating random data [{row}x{col}]...')
start = time.time()
X_train, X_test, y_train, y_test = load_data(shape=(row, col), generated_data=['X_train', 'y_train'])
print('{} sec'.format(time.time()-start))
print('Finished.')

regr = LinearRegression(fit_intercept=True, n_jobs=None, copy_X=False)

print()
print('Starting training...')
start = time.time()
reg = regr.fit(X_train, y_train)
print('{} sec'.format(time.time()-start))
print('Finished.')

print()
print('Starting predicting...')
start = time.time()
reg.predict(X_test)
print('{} sec'.format(time.time()-start))
print('Finished.')
