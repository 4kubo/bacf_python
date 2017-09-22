from argparse import ArgumentParser
from os import environ

home = environ['HOME']

def parse_args():
    parser=ArgumentParser()
    parser.add_argument('--path_to_sequences',
                        default=home + '/data/sequences/OTB2015')
    parser.add_argument('--target_seq',
                        type=str,
                        nargs='+',
                        default=['Soccer'])
    parser.add_argument('--model_name',
                        type=str,
                        default='srdcf')
    parser.add_argument('--run_id',
                        type=str,
                        default='result')
    parser.add_argument('--save_without_showing',
                        action='store_true')

    # Default parameters used in the ICCV 2015 paper
    parser.add_argument('--dim_feature', default=31,
                        help="HOG feature parameters")
    parser.add_argument('--colorspace', default='gray',
                        help="Grayscale feature parameters")
    parser.add_argument('--n_dim', default=1)

    # Global feature parameters


    parser.add_argument('--cell_size', default=4,
                        help='Feature cell size')
    parser.add_argument('--cell_selection_thresh', default=0.75**2,
                        help='Threshold for reducing the cell size in low-resolution cases')

    # Filter parameters
    parser.add_argument('--search_area_shape', default='square',
                        help="The shape of the training/detection window: 'proportional', 'square' or 'fix_padding'")
    parser.add_argument('--search_area_scale', default=5.0,
                        type=float,
                        help="the size of the training/detection area proportional to the target size")
    parser.add_argument('--filter_max_area', default=50**2,
                        help="the size of the training/detection area in feature grid cells")

    # Learning parameters
    parser.add_argument('--learning_rate', default=0.013,
                        help="learning rate")
    parser.add_argument('--output_sigma_factor', default=1./16, type=float,
                        help="standard deviation of the desired correlation output (proportional to target)")

    # Detection parameters
    parser.add_argument('--refinement_iterations', default=1,
                        help="number of iterations used to refine the resulting position in a frame")
    parser.add_argument('--interpolate_response', default=4,
                        help="correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method")
    parser.add_argument('--newton_iterations', default=5,
                        help="number of Newton's iteration to maximize the detection scores")

    # Regularization window parameters
    parser.add_argument('--use_reg_window', default=1,
                        help="whether to use windowed regularization or not")
    parser.add_argument('--reg_window_min', default=0.1,
                        help="the minimum value of the regularization window")
    parser.add_argument('--reg_window_edge', default=3.0,
                        help="the impact of the spatial regularization (value at the target border), depends on the detection size and the feature dimensionality")
    parser.add_argument('--reg_window_power', default=2,
                        help="the degree of the polynomial to use (e.g. 2 is a quadratic window)")
    parser.add_argument('--reg_sparsity_threshold', default=0.05,
                        help="a relative threshold of which DFT coefficients that should be set to zero")
    parser.add_argument('--reg_lambda', default=0.01,
                        help="the weight of the standard (uniform) regularization, only used when use_reg_window == 0")

    # Scale parameters
    parser.add_argument('--n_scales', type=int, default=5)
    parser.add_argument('--scale_step', default=1.01)

    # Debug and visualization
    parser.add_argument('--visualization',
                        action='store_true')
    parser.add_argument('--debug',
                        action="store_true")

    params = parser.parse_args()
    for param in dir(params):
        if not param.startswith('_'):
            print('{0} : {1}'.format(param, getattr(params, param)))
    return params