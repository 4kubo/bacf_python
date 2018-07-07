import os
import sys
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt


class LogManger(object):
    def __init__(self, tracker, base_path_to_save, target_items, elapsed_time=False, env=None,
                 visualization=False, is_detailed=False, is_simplest=False, save_without_showing=False):
        self._tracker = tracker
        self._base_path_to_save = base_path_to_save
        self._elapsed_time = elapsed_time
        self._target_items = target_items
        self._env = env
        self._visualization = visualization
        self._is_detailed = is_detailed
        self._is_simplest = is_simplest
        self._save_without_showing = save_without_showing

        # Set items to report
        for target_item in target_items:
            attr_name = "_{0}s".format(target_item)
            setattr(self, attr_name, [])
        if elapsed_time is True:
            self._e_times = []

        # For a tracking environment in RL
        self._y = []
        if env is not None:
            self._targets = env.get_targets()
            self._n_target = len(self._targets)
            self._aucs = {k: [] for k in self._targets}
            self._ious = []

        if visualization:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)

    def store(self, is_timer=False, **items):
        # Set a timer
        if self._elapsed_time is True and is_timer:
            if not hasattr(self, "_time0"):
                raise AttributeError("Timer is not set. Please set a time")
            time1 = time.time()
            e_time = time1 - self._time0
            self._time0 = time1
            self._e_times.append(e_time)
        # Append each value
        for target_item, value in items.items():
            attr_name = "_{0}s".format(target_item)
            assert hasattr(self, attr_name)
            attr = getattr(self, attr_name)
            attr.append(value)

    def init(self, report_id, model_id):
        self._time0 = time.time()
        self.report_id = report_id
        self.model_id = model_id

    def report(self, report_id, target_items=[], elapsed_time=True):
        texts = ["@{0} ".format(report_id)]
        for target_item in target_items:
            # Tracking accuracy
            if target_item == "auc" or target_item == "iou":
                self._env.set_noa(self._ious)
                _, auc = self._env.get_eval()
                texts += ["AUC : {0:.3}".format(auc)]
            else:
                attr_name = "_{0}s".format(target_item)
                assert(hasattr(self, attr_name))
                attr = getattr(self, attr_name)
                value = np.array(attr).mean()
                text_item = "{0} : {1:.3}".format(target_item, value)
                texts += [text_item]

        # Store immediate IoUs
        if self._env is not None:
            srs, auc = self._env.get_eval()
            texts += ["auc : {0:.3}".format(auc)]
            iou = self._env.get_noa()
            self._ious.extend(iou)

        if self._elapsed_time and elapsed_time:
            fps = (1 / np.array(self._e_times)).mean()
            texts += ["fps : {0:.3}".format(fps)]
        print(", ".join(text for text in texts))

    def save_results(self):
        results = {}
        for target_item in self._target_items:
            attr_name = "_{0}s".format(target_item)
            attr = getattr(self, attr_name)
            results[target_item] = attr

        # Save bbox results to csv
        path_to_csv = "{0}/rect_pos_csv/{1}"\
            .format(self._base_path_to_save, self.report_id)
        if not os.path.exists(path_to_csv):
            os.makedirs(path_to_csv)
            print("Made a directory : {0}".format(path_to_csv))
        csv_file_name = "{0}/{1}_{2}.csv".format(path_to_csv, self.model_id, self.report_id)
        np.savetxt(csv_file_name, np.array(self._rect_poss), delimiter=",")
        print("Saved results to {0}".format(csv_file_name))

        # Save misc results to pkl
        path_to_pkl = "{0}/misc/{1}".format(self._base_path_to_save, self.report_id)
        if not os.path.exists(path_to_pkl):
            os.makedirs(path_to_pkl)
            print("Made a directory : {0}".format(path_to_pkl))
        pkl_file_name = "{0}/{1}_{2}.pkl".format(path_to_pkl, self.model_id, self.report_id)
        with open(pkl_file_name, "w") as f:
            pickle.dump(results, f)
        print("Saved results to {0}".format(pkl_file_name))
        sys.stdout.flush()

    def clear_results(self, target_items=[]):
        # Clear all results
        if len(target_items) is 0:
            for target_item in self._target_items:
                attr_name = "_{0}s".format(target_item)
                setattr(self, attr_name, [])
        # Clear appointed results
        else:
            for target_item in target_items:
                attr_name = "_{0}s".format(target_item)
                if hasattr(self, attr_name):
                    setattr(self, attr_name, [])
                else:
                    print("attribution {0} is not included".format(attr_name))

        if self._env is not None:
            self._ious = []

    def visualize(self):
        if self._save_without_showing:
            save_without_showing = "{0}/images".format(self._base_path_to_save)
        else:
            save_without_showing = self._save_without_showing
        self._tracker.visualise(self.report_id, is_detailed=self._is_detailed,
                                is_simplest=self._is_simplest,
                                save_without_showing=save_without_showing)

    def update_env(self):
        # Update aucs dict
        srs, auc = self._env.get_eval()
        target = self._env.target
        self._aucs[target].append(auc)

        # Mean auc
        valid_key = [key for key in self._aucs.keys() if len(self._aucs[key]) is not 0]
        n_valid = len(valid_key)
        current_mean_auc = sum([self._aucs[key][-1] for key in valid_key]) / n_valid
        self._y.append(current_mean_auc)
        self._current_mean_auc = current_mean_auc
        self._n_valid = n_valid

    def report_env(self):
        print("Current mean auc over {0} / {2} seqs: {1:.3}"
              .format(self._n_valid, self._current_mean_auc, self._n_target))
        if self._visualization:
            self._ax.clear()
            self._ax.plot(self._y)
            self._fig.savefig("plot.pdf")