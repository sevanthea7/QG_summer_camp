import os
from shutil import copy, rmtree
import random

def make_file( path: str ):
    if os.path.exists( path ):
        rmtree( path )
    os.makedirs( path )



random.seed( 0 )
test_rate = 0.1


pre_folder = '../data'
leaf_p = os.path.join( pre_folder, 'leaf_data' )
# assert os.path.exists( leaf_p ), "path '{}' does not exist.".format( leaf_p )


classes = []
for c in os.listdir( leaf_p ):
    if os.path.isdir( os.path.join( leaf_p, c ) ):
        classes.append( c )


train_folder = os.path.join( pre_folder, 'train_photos' )
make_file( train_folder )
for c in classes:
    make_file( os.path.join( train_folder, c ) )

test_folder = os.path.join( pre_folder, 'test_photos' )
make_file( test_folder )
for c in classes:
    make_file( os.path.join( test_folder, c ) )


for c in classes:
    c_path = os.path.join( leaf_p, c )
    images = os.listdir( c_path )
    num = len( images )
    ran_idx = random.sample( images, int( num * test_rate ) )
    print( int( num * test_rate ) )

    for idx, img in enumerate( images ):
        if img in ran_idx:
            img_path = os.path.join( c_path, img )
            new_path = os.path.join( test_folder, c )
            copy( img_path, new_path )
        else:
            img_path = os.path.join( c_path, img )
            new_path = os.path.join( train_folder, c )
            copy( img_path, new_path )

print( 'finished' )