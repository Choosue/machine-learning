% Read Files
file_names = readFileName(2500, "./spamassasin-dataset/easy_ham/cmds");

% Store all the examples into X
X = zeros(25, 1899);

for i = 1:25
    % Extract Features
    file_contents  = readFile(strcat("./spamassasin-dataset/easy_ham/", file_names{1}));
    word_indices = processEmail(file_contents);
    X(i, :)            = emailFeatures(word_indices);

    % Print Stats
    fprintf("Length of feature vector: %d\n", length(X(i, :)));
    fprintf("Number of non-zero entries: %d\n", sum(X(i, :) > 0));
end
save("-ascii", "spamassasinTrain.mat", "X");