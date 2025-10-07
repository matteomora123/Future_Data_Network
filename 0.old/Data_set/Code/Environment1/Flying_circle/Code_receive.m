%Receive distance measurement data frames during circle formation flight in indoor environments
%Environment1-Flying_circle

clc
clear

%Establishing a serial port
s = serial('COM8','InputBufferSize',1024,'terminator','','Timeout',30,'BaudRate',921600,'DataBits',8);
fclose(instrfind);

%% CH2 measurement
for i = 1:30
    fopen(s);
    data_s = fread(s);
    data_hex = dec2hex(data_s,2);
    Hex_root_mat = strcat('I:\Data_set\Raw_data\Environment1\Flying_circle\Flying_point',num2str(i),'\Original_data\data_hex_orgin_ch2.mat');
    data_hex_orgin_ch2 = data_hex;
    save(Hex_root_mat,'data_hex_orgin_ch2');
    fclose(s);
end


%% CH3 measurement
for i = 1:30 
    fopen(s);
    data_s = fread(s);
    data_hex = dec2hex(data_s,2);
    Hex_root_mat = strcat('I:\Data_set\Raw_data\Environment1\Flying_circle\Flying_point',num2str(i),'\Original_data\data_hex_orgin_ch3.mat');
    data_hex_orgin_ch3 = data_hex;
    save(Hex_root_mat,'data_hex_orgin_ch3');
    fclose(s);
end


%% CH4 measurement
for i = 1:30
    fopen(s);
    data_s = fread(s);
    data_hex = dec2hex(data_s,2);
    Hex_root_mat = strcat('I:\Data_set\Raw_data\Environment1\Flying_circle\Flying_point',num2str(i),'\Original_data\data_hex_orgin_ch4.mat');
    data_hex_orgin_ch4 = data_hex;
    save(Hex_root_mat,'data_hex_orgin_ch4');
    fclose(s);
end


%% CH5 measurement
for i = 1:30
    fopen(s);
    data_s = fread(s);
    data_hex = dec2hex(data_s,2);
    Hex_root_mat = strcat('I:\Data_set\Raw_data\Environment1\Flying_circle\Flying_point',num2str(i),'\Original_data\data_hex_orgin_ch5.mat');
    data_hex_orgin_ch5 = data_hex;
    save(Hex_root_mat,'data_hex_orgin_ch5');
    fclose(s);
end


%% Delete all serial ports
delete(instrfindall);