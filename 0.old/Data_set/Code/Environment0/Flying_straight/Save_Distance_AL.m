function [i, j] = Save_Distance_AL(data, i, j)
%   Save the parsed ranging matrix in the official folder of the dataset
%   i: the i-th trajectory point
%   j: the j-th channel, with channel labels of 2,3,4,5


Save_path = strcat('I:\Data_set\Raw_data\Environment0\Flying_straight\Flying_point',num2str(i),'\Distance_Anchor_Label\Dis_anchor_label_ch',num2str(j+1),'.mat');

if j+1 == 2
    Dis_anchor_label_ch2 = data;
    save(Save_path,'Dis_anchor_label_ch2');
end
if j+1 == 3
    Dis_anchor_label_ch3 = data;
    save(Save_path,'Dis_anchor_label_ch3');
end
if j+1 == 4
    Dis_anchor_label_ch4 = data;
    save(Save_path,'Dis_anchor_label_ch4');
end
if j+1 == 5
    Dis_anchor_label_ch5 = data;
    save(Save_path,'Dis_anchor_label_ch5');
end

end