import os
import shutil
import pandas as pd

in_f = '../data/cassava-leaf-disease-classification/train_images'
out_f = '../data/leaf_data'

for i in range( 5 ):
    label_f = os.path.join( out_f, str( i ) )
    os.makedirs( label_f, exist_ok=True )


csv_data = pd.read_csv( '../data/cassava-leaf-disease-classification/train.csv' )

for i, data in csv_data.iterrows():
    img = data[ 'image_id' ]
    label = data[ 'label' ]
    in_path = os.path.join( in_f, img )
    out_path = os.path.join( out_f, str( label ), img )

    if os.path.exists( in_path ):
        shutil.move( in_path, out_path )
        # print( f'{img} -> {out_path}' )
    else:
        print( f'{img} failed' )

print( 'finished' )