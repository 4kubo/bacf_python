from argparse import ArgumentParser



def parse_args():
    parser=ArgumentParser()
    # Configuration
    parser.add_argument('--paths_to_seqs',
                        type=str,
                        nargs='+',
                        default=['sequences'])
    parser.add_argument('--target_seqs',
                        type=str,
                        nargs='+',
                        default=['Bolt'])
    parser.add_argument('--paths_to_test_seqs',
                        type=str,
                        nargs='+',
                        default=['sequences'])
    parser.add_argument('--target_test_seqs',
                        type=str,
                        nargs='+',
                        default=['Bolt'])
    parser.add_argument('--model_name',
                        type=str,
                        default='bacf')
    parser.add_argument('--run_id',
                        type=str,
                        default='result')
    parser.add_argument('--save_without_showing',
                        action='store_true')
    parser.add_argument('--path_to_save',
                        type=str,
                        default="./result")
    parser.add_argument('--path_to_load',
                        type=str,
                        default="")
    parser.add_argument('--gpu_fraction',
                        type=float,
                        default=0.4)
    parser.add_argument('--elapsed_time',
                        action="store_true",
                        help="A flag to measure elapsed time")

    # Default parameters
    parser.add_argument('--dim_feature', default=31,
                        help="HOG feature parameters")
    parser.add_argument('--n_dim', default=1)

    # Global feature parameters
    parser.add_argument('--feature_ratio', type=float, default=4,
                        help='Feature cell size')
    parser.add_argument('--cell_selection_thresh', type=float, default=0.75**2,
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
    parser.add_argument('--learning_rate', type=float, default=0.013,
                        help="learning rate")
    parser.add_argument('--output_sigma_factor', default=1./16, type=float,
                        help="standard deviation of the desired correlation output (proportional to target)")

    # Redetection configurations
    parser.add_argument('--is_redetection', action="store_true",
                        help="If set, redetect the targets after tracking failure")
    parser.add_argument('--is_entire_redection', action="store_true",
                        help="If set, search the targets' location over an entire input frame")
    parser.add_argument('--redetection_search_area_scale', type=float, default=2.0,
                        help="Scale of a searching area for redetection")
    parser.add_argument('--psr_threshold', type=float, default=2.0,
                        help="Threshold of PSR value to redetect")

    # Detection parameters
    parser.add_argument('--refinement_iterations', default=1, type=int,
                        help="number of iterations used to refine the resulting position in a frame")
    parser.add_argument('--interpolate_response', default=4,
                        help="correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method")
    parser.add_argument('--newton_iterations', type=int, default=5,
                        help="number of Newton's iteration to maximize the detection scores")

    # Regularization window parameters
    parser.add_argument('--use_reg_window', default=1,
                        help="whether to use windowed regularization or not")
    parser.add_argument('--reg_window_min', type=float, default=0.1,
                        help="the minimum value of the regularization window")
    parser.add_argument('--reg_window_edge', type=float, default=3.0,
                        help="the impact of the spatial regularization (value at the target border), depends on the detection size and the feature dimensionality")
    parser.add_argument('--reg_window_power', type=float, default=2,
                        help="the degree of the polynomial to use (e.g. 2 is a quadratic window)")
    parser.add_argument('--reg_lambda', type=float, default=0.01,
                        help="the weight of the standard (uniform) regularization, only used when use_reg_window == 0")

    # Scale parameters
    parser.add_argument('--n_scales', type=int, default=5)
    parser.add_argument('--scale_step', type=float, default=1.01)

    # Optimization parameters for ADMM
    parser.add_argument('--admm_lambda', type=float, default=0.01)

    # Configuration for reinforcement learning
    parser.add_argument('--n_episode', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=3)
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--load_model', action="store_true")
    parser.add_argument('--is_hann', action="store_true")

    # Debug and visualization
    parser.add_argument('--visualization', '-v',
                        action='store_true')
    parser.add_argument('--debug',
                        action="store_true")

    return parser