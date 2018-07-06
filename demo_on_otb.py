import traceback
import cv2

from background_aware_correlation_filter import BackgroundAwareCorrelationFilter as BACF
from utils.arg_parse import parse_args
from image_process.feature import get_pyhog
from utils.get_sequence import get_sequence_info, load_image
from utils.report import LogManger

def show_image_with_bbox(image, rect_pos):
    tl = (int(rect_pos[0]), int(rect_pos[1]))
    br = (int(rect_pos[0] + rect_pos[2]), int(rect_pos[1] + rect_pos[3]))
    image_with_bbox = cv2.rectangle(image, tl, br, (255, 0, 0), 3)
    cv2.imshow("image_with_bbox", image_with_bbox)
    cv2.waitKey(1)

if __name__ == "__main__":
    """This demo script runs the BACF tracker"""
    # Parse command line arguments
    parser = parse_args()
    params = parser.parse_args()
    for param in dir(params):
        if not param.startswith('_'):
            print('{0} : {1}'.format(param, getattr(params, param)))

    bacf = BACF(get_pyhog, admm_lambda=params.admm_lambda,
                cell_selection_thresh=params.cell_selection_thresh,
                dim_feature=params.dim_feature,
                filter_max_area=params.filter_max_area,
                feature_ratio=params.feature_ratio,
                interpolate_response=params.interpolate_response,
                learning_rate=params.learning_rate,
                search_area_scale=params.search_area_scale,
                reg_window_power=params.reg_window_power,
                n_scales=params.n_scales,
                newton_iterations=params.newton_iterations,
                output_sigma_factor=params.output_sigma_factor,
                refinement_iterations=params.refinement_iterations,
                reg_lambda=params.reg_lambda,
                reg_window_edge=params.reg_window_edge,
                reg_window_min=params.reg_window_min,
                scale_step=params.scale_step,
                search_area_shape=params.search_area_shape,
                save_without_showing=params.save_without_showing,
                debug=params.debug,
                visualization=params.visualization,
                is_redetection=params.is_redetection,
                redetection_search_area_scale=params.redetection_search_area_scale,
                is_entire_redection=params.is_entire_redection,
                psr_threshold=params.psr_threshold)

    # Arrange path and ground truth label for each sequence
    _, info = \
        get_sequence_info(params.paths_to_seqs, params.target_seqs, params.target_test_seqs)

    seq_names, gt_labels, frames_names, n_frames = info

    # For report and saving tracking results
    log_manager = LogManger(params.path_to_save, ["rect_pos", "psr"],
                            elapsed_time=params.elapsed_time)

    for i, (seq_name, gt_label, image_names, n_frame) in enumerate(zip(seq_names, gt_labels, frames_names, n_frames)):
        print("Current sequence : {}".format(seq_name))
        try:
            images = load_image(image_names)
            rect_pos = gt_label[0, :]

            # Initialise for current images
            patch = bacf.init(images[0], rect_pos)

            log_manager.set_timer()
            log_manager.store(**{"rect_pos": rect_pos})

            # Run BACF
            for i, image in enumerate(images[1:]):
                # Visualization
                # TODO: use visualization function of BACF class
                if params.visualization:
                    image_ = images[i].copy()
                    show_image_with_bbox(image_, rect_pos)

                patch, response = bacf.track(image)
                bacf.train(image)
                _, rect_pos, _, _ = bacf.get_state()

                log_manager.store(is_timer=True, **{"rect_pos": rect_pos, "psr": bacf.psr})
            log_manager.report(seq_name)
            log_manager.save_results(seq_name, "BACF")
            log_manager.clear_results()
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Target {0} is an invalid sequence. So skip".format(seq_name))
            continue