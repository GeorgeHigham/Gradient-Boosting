import numpy as np
import bisect

class GradBoostClassification:
    def __init__(self, x_data, y_data, step_size, data_split):
        self.step_size = step_size
        self.X_train, self.X_test, self.y_train, self.y_test = self.DataProcessing(x_data, y_data, data_split)
        self.current_train_estimate = self.FirstPrediction(self.X_train.shape[0])
        self.splits, self.split_order = self.find_primary_splits(self.X_train, self.y_train)
        self.trees = []

    def log_func(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def DataProcessing(self, x_data, y_data, data_split):
        X_train = x_data[:data_split,:]
        X_test = x_data[data_split:,:]

        y_train = y_data[:data_split]
        y_test = y_data[data_split:]

        return X_train, X_test, y_train, y_test
    
    def FirstPrediction(self, length):
        first_estimation = np.log(np.mean(self.y_train, axis=0)/(1-np.mean(self.y_train, axis=0)))
        estimation = np.array([first_estimation] * length)
        return estimation
    
    def calculate_entropy(self, y):
        if len(set(y)) < 2:
            entropy = 0
        else:
            prob_1 = np.mean(y)
            prob_0 = 1-prob_1
            entropy = -((prob_0*np.log2(prob_0))+(prob_1*np.log2(prob_1)))
        return entropy
    
    # using infogain to find tree splits, slow and high computational cost but finds most effective splits
    # stored for all primary splits so to easily reference
    def find_primary_splits(self, x, y):
        start_entropy = self.calculate_entropy(y)

        split_dict = {}
        split_order = []
        scores_order = []
        
        # going through all features and finding the best split for each
        for i in range(x.shape[1]):
            # getting evenly distributed split samples as is too large otherwise
            splits = np.unique(x[:, i])
            indices = np.linspace(0, len(splits) - 1, min(len(splits), 50), dtype=int)
            split_samples = splits[indices]
            feature_best = -np.inf
            for split in split_samples:
            
                left_indices = []
                right_indices = []

                for ind in range(len(x[:,i])):
                    if x[:,i][ind] <= split:
                        left_indices.append(ind)
                    else:
                        right_indices.append(ind)
        
                left_binary_array = y[left_indices]
                right_binary_array = y[right_indices]
            

                left_prop = len(left_binary_array) / len(y)
                right_prop = len(right_binary_array) / len(y)


                infogain = start_entropy - (left_prop * self.calculate_entropy(left_binary_array)) - (right_prop * self.calculate_entropy(right_binary_array))
                
                if infogain > feature_best:
                    feature_best = infogain
                    best_split = split
            index = bisect.bisect_left(scores_order, feature_best)
            split_order.insert(index, i)
            scores_order.insert(index, feature_best)
            split_dict[i] = best_split
        return split_dict, split_order
    
    # getting best splits for the current node
    def current_split(self, x, y):
        start_entropy = self.calculate_entropy(y)

        best = -np.inf
        best_split = 0
        best_split_value = -np.inf

        for i in range(x.shape[1]):
            splits = np.unique(x[:, i])
            indices = np.linspace(0, len(splits) - 1, min(len(splits), 50), dtype=int)
            split_samples = splits[indices]
            for split in split_samples:
            
                left_indices = []
                right_indices = []

                for ind in range(len(x[:,i])):
                    if x[:,i][ind] <= split:
                        left_indices.append(ind)
                    else:
                        right_indices.append(ind)
        
                left_binary_array = y[left_indices]
                right_binary_array = y[right_indices]
            

                left_prop = len(left_binary_array) / len(y)
                right_prop = len(right_binary_array) / len(y)


                infogain = start_entropy - (left_prop * self.calculate_entropy(left_binary_array)) - (right_prop * self.calculate_entropy(right_binary_array))
                
                if infogain > best:
                    best = infogain
                    best_split = i
                    best_split_value = split

        return best_split, best_split_value
    
    def store_trees(self, primary_indicator, subsequent_indicator_1, subsequent_indicator_2, primary_indicator_value, subsequent_indicator_1_value, subsequent_indicator_2_value, residual_list):
        tree = [primary_indicator, subsequent_indicator_1, subsequent_indicator_2, primary_indicator_value, subsequent_indicator_1_value, subsequent_indicator_2_value, residual_list]
        self.trees.append(tree)

    def current_tree_update(self):
        # best performance came from testing in order of infogain
        primary_indicator = self.split_order[0]
        self.split_order = self.split_order[1:] + [primary_indicator]
        primary_indicator_value = self.splits[primary_indicator]

        primary_indicator_condition = self.X_train[:, primary_indicator] >= primary_indicator_value
        box_1 = np.where(primary_indicator_condition)[0]
        box_2 = np.where(~primary_indicator_condition)[0]

        # same subsequent features are chosen if primary split is repeated, so trees beyond 30 provide no new information but may affect performance
        # did try sampling more combinations of subsequent features for same primary feature, but performance did not improve - likely due to trees being too shallow to capture further complexity
        subsequent_indicator_1, subsequent_indicator_1_value = self.current_split(self.X_train[box_1], self.y_train[box_1])
        subsequent_indicator_2, subsequent_indicator_2_value = self.current_split(self.X_train[box_2], self.y_train[box_2])

        subsequent_indicator_1_condition = self.X_train[:, subsequent_indicator_1] >= subsequent_indicator_1_value
        subsequent_indicator_2_condition = self.X_train[:, subsequent_indicator_2] >= subsequent_indicator_2_value

        box_left = np.where(primary_indicator_condition & subsequent_indicator_1_condition)[0]
        box_middle_left = np.where(primary_indicator_condition & ~subsequent_indicator_1_condition)[0]
        box_middle_right = np.where(~primary_indicator_condition & subsequent_indicator_2_condition)[0]
        box_right = np.where(~primary_indicator_condition & ~subsequent_indicator_2_condition)[0]

        box_list = [box_left, box_middle_left, box_middle_right, box_right]
        residual_list =[]
        for box in box_list:
            if len(box) == 0:
                residual_list.append(0)
                continue
            observed = self.y_train[box]
            predicted = self.current_train_estimate[box]
            residuals = observed - predicted
            resid_gamma = np.sum(residuals)/np.sum(predicted*(1-predicted))
            residual_list.append(resid_gamma)
            log_odds_prediction = self.current_train_estimate[box] + (self.step_size * resid_gamma)
            self.current_train_estimate[box] = self.log_func(log_odds_prediction)

        self.store_trees(primary_indicator, subsequent_indicator_1, subsequent_indicator_2, primary_indicator_value, subsequent_indicator_1_value, subsequent_indicator_2_value, residual_list)



    def test_all_trees(self, old_trees, x_data):
        # using training initial esimate
        self.current_test_estimate = self.FirstPrediction(x_data.shape[0])

        for tree in old_trees:
            primary_indicator, subsequent_indicator_1, subsequent_indicator_2 = tree[0:3]
            primary_indicator_value, subsequent_indicator_1_value, subsequent_indicator_2_value = tree[3:6]
            residual_list = tree[6]

            primary_indicator_condition = x_data[:, primary_indicator] >= primary_indicator_value
            subsequent_indicator_1_condition = x_data[:, subsequent_indicator_1] >= subsequent_indicator_1_value
            subsequent_indicator_2_condition = x_data[:, subsequent_indicator_2] >= subsequent_indicator_2_value

            box_left = np.where(primary_indicator_condition & subsequent_indicator_1_condition)[0]
            box_middle_left = np.where(primary_indicator_condition & ~subsequent_indicator_1_condition)[0]
            box_middle_right = np.where(~primary_indicator_condition & subsequent_indicator_2_condition)[0]
            box_right = np.where(~primary_indicator_condition & ~subsequent_indicator_2_condition)[0]

            box_list = [box_left, box_middle_left, box_middle_right, box_right]
            for ind, box in enumerate(box_list):
                if box.size == 0:
                    continue
                # changing test current estimate based on the residuals from those in the same boxes in training
                model_est_residual = residual_list[ind]
                log_odds_prediction = self.current_test_estimate[box] + (self.step_size * model_est_residual)
                self.current_test_estimate[box] = self.log_func(log_odds_prediction)
    


        
    
    