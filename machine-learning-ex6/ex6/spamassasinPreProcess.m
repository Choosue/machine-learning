% File Path
file_path = "./spamassasin-dataset/easy_ham/";
save_name = "easyham.mat";

% Number of Examples
m = 2500;

% Read File Names
file_names = readFileName(m, strcat(file_path, "cmds"));

dots = 12;

for i = 1:m
    file_contents = readFile(strcat(file_path, file_names{i}));
    word_array    = processEmailSilent(file_contents);
    save(strcat(strcat(file_path, "processed_email/"), file_names{i}), "word_array");
    
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