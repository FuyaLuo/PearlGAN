clear all;
clc;


night_IR_folder = 'Your NTIR directory';
dir_file = dir(fullfile(night_IR_folder, '*.png'));
day_file_names = {dir_file.name};

des_txt_path = 'Your txt result storage directory';

file_cell_list = cell(1, 2);
file_cell_list{1, 1} = night_IR_folder;
file_cell_list{1, 2} = 'Your translation result directory';


if ~exist(des_txt_path, 'dir'), mkdir(des_txt_path);end

for k = 2: 2
    file_path = file_cell_list{1, k};
    file_path_split = strsplit(file_path, '/');
    methods_name = file_path_split{1, end - 1};
    txt_file_name = [ methods_name, '.txt'];
    fid = fopen(fullfile(des_txt_path, txt_file_name), 'a+');
    while fid ==-1
        fid = fopen(fullfile(des_txt_path, txt_file_name), 'a+');
    end
    fprintf('Procesing Methods is %s\n', methods_name);
    AP_array = zeros(1, 99);
    for j = 1:99

        high_th = j * 0.01;
        low_th = high_th * 0.5;
        fprintf('Procesing Threshold is %.2f\n', high_th);

        precise_ratio = 0.0;
        cnt = 0;
        for i = 1:length(day_file_names)
            img_name = day_file_names{1, i};
            IR_img_file = [night_IR_folder, img_name];
            vis_img_file = [file_path, img_name];
            IR_img = imread(IR_img_file);
            vis_img = imread(vis_img_file);

            IR_edge = edge(IR_img, 'canny', [low_th, high_th]);
            vis_edge = edge(rgb2gray(vis_img), 'canny', [low_th, high_th]);
            if sum(sum(double(IR_edge))) > 0
                cnt = cnt + 1;
                temp_precise_ratio = sum(sum(double(vis_edge) .* double(IR_edge))) / sum(sum(double(IR_edge)));
            else
                temp_precise_ratio = 0;
            end
            
            precise_ratio = precise_ratio + temp_precise_ratio;

        end
        final_precise_ratio = precise_ratio / cnt;
        fprintf(fid,'%.2f %.6f\n',high_th, final_precise_ratio);
        AP_array(1, j) = final_precise_ratio;
    end
    st = fclose(fid);
    while st==-1
        st = fclose(fid);
    end
end

mean_AP = sum(AP_array) / 99;
fprintf('AP under different threshold is %.2f\n', mean_AP);