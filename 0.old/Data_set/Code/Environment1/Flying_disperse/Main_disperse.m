% Environment1-Flying_disperse

clc
clear
close all

Envir_Flag = 1;
Flying_Flag = '10';
MSE_All = [];
Data_hex_orgin = [];
Distance_Anchor_Label = zeros(8, 7, 30);
Distance_Label_Label = zeros(7, 7, 30);
Position_Label = zeros(7, 3, 30);
% Accessing the results of four measurement channels
for j = 1:4
    % Accessing trajectory point1 to point9
    for i = 1:9
        % Read the original ranging data frame
        Data_hex = Get_range_frame(i, j);
        Data_hex_orgin = [Data_hex_orgin; Data_hex];
        % Obtain the ranging matrix based on data frame parsing
        Distance = Code_recode(Data_hex);
        Distance_Anchor_Label(:,:,i) = Distance;
        % Save ranging matrix
        [~,~] = Save_Distance_AL(Distance, i, j);
        % Obtain the reference base station position matrix in this scenario
        Anchor = Get_Anchor(Envir_Flag);
        % Calculate preliminary position based on distance measurement matrix
        Position = Code_position(Anchor,Distance);
        Position_Label(:,:,i) = Position;
        % Save the calculated position matrix
        [~,~] = Save_Position(Position, i, j);
        % Calculate the distance between UAV nodes
        Distance_label_label = Distance_L_L_comput(Position);
        Distance_Label_Label(:,:,i) =  Distance_label_label;
        % Save the distance matrix between UAV nodes
        [~,~] = Save_Distance_LL(Distance_label_label, i, j);
        % Obtain the pre-set trajectory position of the corresponding flight formation in the corresponding scene
        Position_true = Get_Position_true(Flying_Flag,i);
        % Positioning accuracy calculation
        Dis_Mse = Position_Error_process(Position,Position_true);
        % Accumulated positioning error
        MSE_All = cat(2, MSE_All, Dis_Mse');
    end

    % Accessing trajectory point10 to point30
    for i = 10:30
        % Read the original ranging data frame
        Data_hex = Get_range_frame(i, j);
        Data_hex_orgin = [Data_hex_orgin; Data_hex];
        % Obtain the ranging matrix based on data frame parsing
        Distance = Code_recode(Data_hex);
        Distance_Anchor_Label(:,:,i) = Distance;
        % Save ranging matrix
        [~,~] = Save_Distance_AL(Distance, i, j);
        % Obtain the reference base station position matrix in this scenario
        Anchor = Get_Anchor(Envir_Flag);
        % Calculate preliminary position based on distance measurement matrix
        Position = Code_position(Anchor,Distance);
        Position_Label(:,:,i) = Position;
        % Save the calculated position matrix
        [~,~] = Save_Position(Position, i, j);
        % Calculate the distance between UAV nodes
        Distance_label_label = Distance_L_L_comput(Position);
        Distance_Label_Label(:,:,i) =  Distance_label_label;
        % Save the distance matrix between UAV nodes
        [~,~] = Save_Distance_LL(Distance_label_label, i, j);
        % Obtain the pre-set trajectory position of the corresponding flight formation in the corresponding scene
        Position_true = Get_Position_true(Flying_Flag,i);
        % Positioning accuracy calculation
        Dis_Mse = Position_Error_process(Position,Position_true);
        % Accumulated positioning error
        MSE_All = cat(2, MSE_All, Dis_Mse');

       
    end

    if j == 1
        Root_origin = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Original_data\data_hex_orgin_ch',num2str(j+1),'.mat');
        data_hex_orgin_ch2 = Data_hex_orgin;
        save(Root_origin,'data_hex_orgin_ch2');

        Root_Distance_Anchor_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Anchor_Label\Distance_anchor_label_ch',num2str(j+1),'.mat');
        Distance_anchor_label_ch2 = Distance_Anchor_Label;
        save(Root_Distance_Anchor_Label,'Distance_anchor_label_ch2');

        Root_Distance_Label_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Label_Label\Distance_label_label_ch',num2str(j+1),'.mat');
        Distance_label_label_ch2 = Distance_Label_Label;
        save(Root_Distance_Label_Label,'Distance_label_label_ch2');

        Root_Position_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Position_Label\position_label_ch',num2str(j+1),'.mat');
        position_label_ch2 = Position_Label;
        save(Root_Position_Label,'position_label_ch2');
    end

    if j == 2
        Root_origin = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Original_data\data_hex_orgin_ch',num2str(j+1),'.mat');
        data_hex_orgin_ch3 = Data_hex_orgin;
        save(Root_origin,'data_hex_orgin_ch3');

        Root_Distance_Anchor_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Anchor_Label\Distance_anchor_label_ch',num2str(j+1),'.mat');
        Distance_anchor_label_ch3 = Distance_Anchor_Label;
        save(Root_Distance_Anchor_Label,'Distance_anchor_label_ch3');

        Root_Distance_Label_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Label_Label\Distance_label_label_ch',num2str(j+1),'.mat');
        Distance_label_label_ch3 = Distance_Label_Label;
        save(Root_Distance_Label_Label,'Distance_label_label_ch3');

        Root_Position_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Position_Label\position_label_ch',num2str(j+1),'.mat');
        position_label_ch3 = Position_Label;
        save(Root_Position_Label,'position_label_ch3');
    end

    if j == 3
        Root_origin = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Original_data\data_hex_orgin_ch',num2str(j+1),'.mat');
        data_hex_orgin_ch4 = Data_hex_orgin;
        save(Root_origin,'data_hex_orgin_ch4');

        Root_Distance_Anchor_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Anchor_Label\Distance_anchor_label_ch',num2str(j+1),'.mat');
        Distance_anchor_label_ch4 = Distance_Anchor_Label;
        save(Root_Distance_Anchor_Label,'Distance_anchor_label_ch4');

        Root_Distance_Label_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Label_Label\Distance_label_label_ch',num2str(j+1),'.mat');
        Distance_label_label_ch4 = Distance_Label_Label;
        save(Root_Distance_Label_Label,'Distance_label_label_ch4');

        Root_Position_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Position_Label\position_label_ch',num2str(j+1),'.mat');
        position_label_ch4 = Position_Label;
        save(Root_Position_Label,'position_label_ch4');
    end

    if j == 4
        Root_origin = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Original_data\data_hex_orgin_ch',num2str(j+1),'.mat');
        data_hex_orgin_ch5 = Data_hex_orgin;
        save(Root_origin,'data_hex_orgin_ch5');

        Root_Distance_Anchor_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Anchor_Label\Distance_anchor_label_ch',num2str(j+1),'.mat');
        Distance_anchor_label_ch5 = Distance_Anchor_Label;
        save(Root_Distance_Anchor_Label,'Distance_anchor_label_ch5');

        Root_Distance_Label_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Distance_Label_Label\Distance_label_label_ch',num2str(j+1),'.mat');
        Distance_label_label_ch5 = Distance_Label_Label;
        save(Root_Distance_Label_Label,'Distance_label_label_ch5');

        Root_Position_Label = strcat('I:\Data_set\Raw_data\Environment1\Flying_disperse\Position_Label\position_label_ch',num2str(j+1),'.mat');
        position_label_ch5 = Position_Label;
        save(Root_Position_Label,'position_label_ch5');
    end

end

H_Straight = histogram(MSE_All./100,200,'FaceColor',[0.51373, 0.43529, 1],'EdgeColor',[0.51373, 0.43529, 1]);
H_Straight.Normalization = 'probability';
title('Positioning Error [m]:Indoor Disperse');
xlabel('Positioning Error [m]');
ylabel('Probability');
xticks([0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5]);
xlim([0.0 6.5]);
ylim([0 0.04]);
legend('Flying Disperse');
grid on
