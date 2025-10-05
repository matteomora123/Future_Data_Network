function [i, j] = Save_data(data, i, j)
%   Save the spliced data frames in the official folder of the dataset
%   i: the i-th trajectory point
%   j: the j-th channel, with channel labels of 2,3,4,5

Save_path = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Flying_point',num2str(i),'\Original_data\data_hex_orgin_ch',num2str(j+1),'.mat');

if j+1 == 2
    data_hex_orgin_ch2 = data;
    save(Save_path,'data_hex_orgin_ch2');
    
end
if j+1 == 3
    data_hex_orgin_ch3 = data;
    save(Save_path,'data_hex_orgin_ch3');
end
if j+1 == 4
    data_hex_orgin_ch4 = data;
    save(Save_path,'data_hex_orgin_ch4');
end
if j+1 == 5
    data_hex_orgin_ch5 = data;
    save(Save_path,'data_hex_orgin_ch5');
end

end