function [Data_hex] = Get_range_frame(i,j)
% Obtain the corresponding data frames according to the trajectory point number i and channel number j


Hex_path = strcat('I:\Data_set\Raw_data\Environment0\Flying_climb\Flying_point',num2str(i),'\Original_data\');
Dir = dir([Hex_path,'*.mat']);    
Data_hex = load([Hex_path,Dir(j).name]);
if j+1 == 2
    Data_hex = Data_hex.data_hex_orgin_ch2;
end
if j+1 == 3
    Data_hex = Data_hex.data_hex_orgin_ch3;
end    
if j+1 == 4
    Data_hex = Data_hex.data_hex_orgin_ch4;
end    
if j+1 == 5
    Data_hex = Data_hex.data_hex_orgin_ch5;
end    
end