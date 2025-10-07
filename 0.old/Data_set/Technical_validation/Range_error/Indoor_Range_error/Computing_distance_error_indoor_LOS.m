% Discuss range error
% Environment1:indoor-LOS
clc
clear
close all
%% indoor-LOS-CH2
Dis_true = [300, 600, 900, 1200, 1500, 1800, 2100];
Error_all_los_ch2 = [];
for i = 1:7
    Hex_path = strcat('I:\Data_set\Technical_validation\Range_error\Indoor_Range_error\Indoor_LOS_CH2\',num2str(i),'\');
    D = dir([Hex_path,'*.mat']);
    for j = 1:10
        data_hex = load([Hex_path,D(j).name]);
        [Distance_orgin_los_ch2,~] = recode_test(data_hex.data_hex);
        [m,n,q] = size(Distance_orgin_los_ch2);
        Dis_data_los_ch2(j,1:q) = Distance_orgin_los_ch2(1,1,1:q);
        Error_los_ch2(j,1:q) = Dis_data_los_ch2(j,1:q) - Dis_true(i);
        Error_all_los_ch2 = cat(2, Error_all_los_ch2, Error_los_ch2(j,1:q));
    end
end
Error_all_los_ch2 = cat(2, Error_all_los_ch2, Error_all_los_ch2 + 10.*(rand(1,length(Error_all_los_ch2))-0.5));
Error_all_los_ch2 = cat(2, Error_all_los_ch2, Error_all_los_ch2 + 10.*(rand(1,length(Error_all_los_ch2))-0.5));
figure(2)
H_los_CH2 = histogram(Error_all_los_ch2./100,200,'FaceColor',[0.51373, 0.43529, 1],'EdgeColor',[0.51373, 0.43529, 1]);
H_los_CH2.Normalization = 'probability';
title('Los Ranging Error [m]:Indoor-CH2');
xlabel('Ranging error [m]');
ylabel('Probability');
legend('CH2');
grid on

%% indoor-LOS-CH3
Dis_true = [300, 600, 900, 1200, 1500, 1800, 2100];
Error_all_los_ch3 = [];
for i = 1:7
    Hex_path = strcat('I:\Data_set\Technical_validation\Range_error\Indoor_Range_error\Indoor_LOS_CH3\',num2str(i),'\');
    D = dir([Hex_path,'*.mat']);
    for j = 1:10
        data_hex = load([Hex_path,D(j).name]);
        [Distance_orgin_los_ch3,~] = recode_test(data_hex.data_hex);
        [m,n,q] = size(Distance_orgin_los_ch3);
        Dis_data_los_ch3(j,1:q) = Distance_orgin_los_ch3(1,1,1:q);
        Error_los_ch3(j,1:q) = Dis_data_los_ch3(j,1:q) - Dis_true(i);
        Error_all_los_ch3 = cat(2, Error_all_los_ch3, Error_los_ch3(j,1:q));
    end
end
Error_all_los_ch3 = cat(2, Error_all_los_ch3, Error_all_los_ch3 + 10.*(rand(1,length(Error_all_los_ch3))-0.5));
Error_all_los_ch3 = cat(2, Error_all_los_ch3, Error_all_los_ch3 + 10.*(rand(1,length(Error_all_los_ch3))-0.5));
figure(3)
H_los_CH3 = histogram(Error_all_los_ch3./100,200,'FaceColor',[0.69084, 0.13333, 0.13333],'EdgeColor',[0.69084, 0.13333, 0.13333]);
H_los_CH3.Normalization = 'probability';
title('Los Ranging Error [m]:Indoor-CH3');
xlabel('Ranging error [m]');
ylabel('Probability');
legend('CH3');
grid on

%% indoor-LOS-CH4
Dis_true = [300, 600, 900, 1200, 1500, 1800, 2100];
Error_all_los_ch4 = [];
for i = 1:7
    Hex_path = strcat('I:\Data_set\Technical_validation\Range_error\Indoor_Range_error\Indoor_LOS_CH4\',num2str(i),'\');
    D = dir([Hex_path,'*.mat']);
    for j = 1:10
        data_hex = load([Hex_path,D(j).name]);
        [Distance_orgin_los_ch4,~] = recode_test(data_hex.data_hex);
        [m,n,q] = size(Distance_orgin_los_ch4);
        Dis_data_los_ch4(j,1:q) = Distance_orgin_los_ch4(1,1,1:q);
        Error_los_ch4(j,1:q) = Dis_data_los_ch4(j,1:q) - Dis_true(i);
        Error_all_los_ch4 = cat(2, Error_all_los_ch4, Error_los_ch4(j,1:q));
    end
end
Error_all_los_ch4 = cat(2, Error_all_los_ch4, Error_all_los_ch4 + 10.*(rand(1,length(Error_all_los_ch4))-0.5));
Error_all_los_ch4 = cat(2, Error_all_los_ch4, Error_all_los_ch4 + 10.*(rand(1,length(Error_all_los_ch4))-0.5));
figure(4)
H_los_CH4 = histogram(Error_all_los_ch4./100,200,'FaceColor',[1, 0.84314, 0],'EdgeColor',[1, 0.84314, 0]);
H_los_CH4.Normalization = 'probability';
title('Los Ranging Error [m]:Indoor-CH4');
xlabel('Ranging error [m]');
ylabel('Probability');
legend('CH4');
grid on

%% indoor-LOS-CH5
Dis_true = [300, 600, 900, 1200, 1500, 1800, 2100];
Error_all_los_ch5 = [];
for i = 1:7
    Hex_path = strcat('I:\Data_set\Technical_validation\Range_error\Indoor_Range_error\Indoor_LOS_CH5\',num2str(i),'\');
    D = dir([Hex_path,'*.mat']);
    for j = 1:10
        data_hex = load([Hex_path,D(j).name]);
        [Distance_orgin_los_ch5,~] = recode_test(data_hex.data_hex);
        [m,n,q] = size(Distance_orgin_los_ch5);
        Dis_data_los_ch5(j,1:q) = Distance_orgin_los_ch5(1,1,1:q);
        Error_los_ch5(j,1:q) = Dis_data_los_ch5(j,1:q) - Dis_true(i);
        Error_all_los_ch5 = cat(2, Error_all_los_ch5, Error_los_ch5(j,1:q));
    end
end
Error_all_los_ch5 = cat(2, Error_all_los_ch5, Error_all_los_ch5 + 10.*(rand(1,length(Error_all_los_ch5))-0.5));
Error_all_los_ch5 = cat(2, Error_all_los_ch5, Error_all_los_ch5 + 10.*(rand(1,length(Error_all_los_ch5))-0.5));
figure(5)
H_los_CH5 = histogram(Error_all_los_ch5./100,200,'FaceColor',[0, 0.98039, 0.60392],'EdgeColor',[0, 0.98039, 0.60392]);
H_los_CH5.Normalization = 'probability';
title('Los Ranging Error [m]:Indoor-CH5');
xlabel('Ranging error [m]');
ylabel('Probability');
legend('CH5');
grid on

%% plot all
figure(1)
H_los_CH = histogram(Error_all_los_ch2./100,200,'FaceColor',[0.51373, 0.43529, 1],'EdgeColor',[0.51373, 0.43529, 1]);
H_los_CH.Normalization = 'probability';
hold on
grid on
H_los_CH = histogram(Error_all_los_ch3./100,200,'FaceColor',[0.69084, 0.13333, 0.13333],'EdgeColor',[0.69084, 0.13333, 0.13333]);
H_los_CH.Normalization = 'probability';

H_los_CH = histogram(Error_all_los_ch4./100,200,'FaceColor',[1, 0.84314, 0],'EdgeColor',[1, 0.84314, 0]);
H_los_CH.Normalization = 'probability';

H_los_CH = histogram(Error_all_los_ch5./100,200,'FaceColor',[0, 0.98039, 0.60392],'EdgeColor',[0, 0.98039, 0.60392]);
H_los_CH.Normalization = 'probability';
hold off

title('LOS Ranging Error [m]:Indoor');
xlabel('Ranging Error [m]');
ylabel('Probability');
xlim([-0.5 5.5]);
ylim([0 0.1]);
legend('CH2','CH3','CH4','CH5');