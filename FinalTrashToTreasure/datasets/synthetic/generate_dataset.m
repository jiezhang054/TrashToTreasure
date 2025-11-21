rng(2002); 

sample_size = 3000;


view1 = randn(sample_size, 200);


view2 = randn(sample_size, 100);


combined_view = [view1, view2];


coefficients = randn(300, 1); 
linear_relation = combined_view * coefficients;


num_classes = 10;
samples_per_class = round(sample_size / num_classes);
[~, sorted_indices] = sort(linear_relation);
gt_labels(sorted_indices) = repelem(1:num_classes, samples_per_class);


train_ratio = 0.7;
test_ratio = 0.3;

num_train = round(sample_size * train_ratio);
num_test = sample_size - num_train;


shuffled_indices = randperm(sample_size);
shuffled_combined_view = combined_view(shuffled_indices, :);
shuffled_gt_labels = gt_labels(shuffled_indices);


X1_train = shuffled_combined_view(1:num_train, 1:200);
X1_test = shuffled_combined_view(num_train+1:end, 1:200);

X2_train = shuffled_combined_view(1:num_train, 201:end);
X2_test = shuffled_combined_view(num_train+1:end, 201:end);

gt_train = shuffled_gt_labels(1:num_train);
gt_test = shuffled_gt_labels(num_train+1:end);


save('synthetic.mat', 'X1_train', 'X1_test', 'X2_train',  'X2_test', 'gt_train', 'gt_test');

disp('dataset saved');
