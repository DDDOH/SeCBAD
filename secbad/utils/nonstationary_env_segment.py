import numpy as np
import os


class NonstationaryEnvSegment():
    """Generate segment length for non stationary envs.
    """

    def __init__(self, segment_len_mean, segment_len_std, segment_len_min, sample_func, store_dir, vis_trainingstep):
        self.segment_len_mean = segment_len_mean
        self.segment_len_std = segment_len_std
        self.segment_len_min = segment_len_min
        self.next_segment_start = int(max(np.random.normal(
            self.segment_len_mean, self.segment_len_std), self.segment_len_min))
        self.step = 0  # current trajectory step

        self.sample_func = sample_func
        self.store_dir = store_dir
        # list of training steps for visualization
        self.vis_trainingstep = vis_trainingstep

        if sample_func is not None:
            self.store_evaluate_task()

    def sample_traj_task(self):
        raise NotImplemented

    def store_evaluate_task(self):
        # if file does not exists, sample tasks
        if not os.path.exists(self.store_dir + '/evaluate_tasks.npy'):
            # key: trainingstep, value: task
            evaluate_tasks = {}
            for i in range(len(self.vis_trainingstep)):
                evaluate_tasks[self.vis_trainingstep[i]] = self.sample_func()
            # store the task for evaluation, one file per env
            print('sample tasks for evaluation and saved to ' +
                  self.store_dir + '/evaluate_tasks.npy')
            np.save(self.store_dir + '/evaluate_tasks.npy', evaluate_tasks)

    @staticmethod
    def get_task(step_idx, store_dir):
        # load npy file each evaluation step for now
        evaluate_tasks = np.load(
            store_dir + '/evaluate_tasks.npy', allow_pickle=True).item()
        vis_trainingstep = np.array(list(evaluate_tasks.keys()))
        task_id = vis_trainingstep[np.argmax(vis_trainingstep <= step_idx)]
        return evaluate_tasks[task_id]

    # @staticmethod
    # def read_evaluate_task(store_dir):
    #     evaluate_tasks = np.load(
    #         store_dir + '/evaluate_tasks.npy', allow_pickle=True).item()
    #     return evaluate_tasks

    def whether_new_segment(self, trainingstep):
        if self.step == self.next_segment_start:
            self.next_segment_start += int(max(np.random.normal(
                self.segment_len_mean, self.segment_len_std), self.segment_len_min))
            return True
        else:
            return False
