function [i, j] = Save_Position(data, i, j)
%   Save the calculated position matrix in the official folder of the dataset
%   i: the i-th trajectory point
%   j: the j-th channel, with channel labels of 2,3,4,5

Save_path = strcat('I:\Data_set\Raw_data\Environment0\Flying_climb\Flying_point',num2str(i),'\Position_Label\Position_label_ch',num2str(j+1),'.mat');

if j+1 == 2
    Position_label_ch2 = data;
    save(Save_path,'Position_label_ch2');
end
if j+1 == 3
    Position_label_ch3 = data;
    save(Save_path,'Position_label_ch3');
end
if j+1 == 4
    Position_label_ch4 = data;
    save(Save_path,'Position_label_ch4');
end
if j+1 == 5
   Position_label_ch5 = data;
    save(Save_path,'Position_label_ch5');
end

end