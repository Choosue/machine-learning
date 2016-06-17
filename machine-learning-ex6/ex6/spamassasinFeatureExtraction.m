% File Path
file_path = "./spamassasin-dataset/easy_ham/";
save_name = "easyham.mat";

% Number of Examples
m = 2;

% Read Files
file_names = readFileName(m, strcat(file_path, "cmds"));

% Store all the examples into X
X = zeros(m, 1899);
y = zeros(m, 1);

dots = 12;

for i = 1:m
    file_contents = readFile(strcat(file_path, file_names{i}));
    word_indices  = processEmailSilent(file_contents);
    X(i, :)       = emailFeatures(word_indices);
    
    fprintf('.');
    dots = dots + 1;
    if dots > 78
       dots = 0;
       fprintf('\n');
    end
    if exist('OCTAVE_VERSION')
       fflush(stdout);
    end
end
fprintf(' Done! \n\n');

save(save_name, "X", "y");