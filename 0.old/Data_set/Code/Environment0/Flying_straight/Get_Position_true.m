function [position_true] = Get_Position_true(Flying_Flag,i)
%   First, read the flight trajectory file of the corresponding flight formation in the Guiying environment, and then extract the position coordinate matrix of the corresponding trajectory points according to the marker positions
%   Flag='00':outdoor-straight;
%   Flag='01':outdoor-climb，
%   Flag='10':indoor-disperse，
%   Flag='11':indoor-circle
%   i:Number of trajectory points for a single flying formation

Position_true = zeros(3,7);
if Flying_Flag == '00'
    Position_UAV1 = load('I:\Data_set\Environment0\Flying_path\Flying_straight\Position_UAV1.mat');
    Position_UAV2 = load('I:\Data_set\Environment0\Flying_path\Flying_straight\Position_UAV2.mat');
    Position_UAV3 = load('I:\Data_set\Environment0\Flying_path\Flying_straight\Position_UAV3.mat');
    Position_UAV4 = load('I:\Data_set\Environment0\Flying_path\Flying_straight\Position_UAV4.mat');
    Position_UAV5 = load('I:\Data_set\Environment0\Flying_path\Flying_straight\Position_UAV5.mat');
    Position_UAV6 = load('I:\Data_set\Environment0\Flying_path\Flying_straight\Position_UAV6.mat');
    Position_UAV7 = load('I:\Data_set\Environment0\Flying_path\Flying_straight\Position_UAV7.mat');
    Position_true(:,1) = Position_UAV1.Position_UAV1(:,i);
    Position_true(:,2) = Position_UAV2.Position_UAV2(:,i);
    Position_true(:,3) = Position_UAV3.Position_UAV3(:,i);
    Position_true(:,4) = Position_UAV4.Position_UAV4(:,i);
    Position_true(:,5) = Position_UAV5.Position_UAV5(:,i);
    Position_true(:,6) = Position_UAV6.Position_UAV6(:,i);
    Position_true(:,7) = Position_UAV7.Position_UAV7(:,i);
elseif Flying_Flag == '01'
    Position_UAV1 = load('I:\Data_set\Environment0\Flying_path\Flying_climb\Position_UAV1.mat');
    Position_UAV2 = load('I:\Data_set\Environment0\Flying_path\Flying_climb\Position_UAV2.mat');
    Position_UAV3 = load('I:\Data_set\Environment0\Flying_path\Flying_climb\Position_UAV3.mat');
    Position_UAV4 = load('I:\Data_set\Environment0\Flying_path\Flying_climb\Position_UAV4.mat');
    Position_UAV5 = load('I:\Data_set\Environment0\Flying_path\Flying_climb\Position_UAV5.mat');
    Position_UAV6 = load('I:\Data_set\Environment0\Flying_path\Flying_climb\Position_UAV6.mat');
    Position_UAV7 = load('I:\Data_set\Environment0\Flying_path\Flying_climb\Position_UAV7.mat');
    Position_true(:,1) = Position_UAV1.Position_UAV1(:,i);
    Position_true(:,2) = Position_UAV2.Position_UAV2(:,i);
    Position_true(:,3) = Position_UAV3.Position_UAV3(:,i);
    Position_true(:,4) = Position_UAV4.Position_UAV4(:,i);
    Position_true(:,5) = Position_UAV5.Position_UAV5(:,i);
    Position_true(:,6) = Position_UAV6.Position_UAV6(:,i);
    Position_true(:,7) = Position_UAV7.Position_UAV7(:,i);
elseif Flying_Flag == '10'
    Position_UAV1 = load('I:\Data_set\Environment1\Flying_path\Flying_circle\Position_UAV1.mat');
    Position_UAV2 = load('I:\Data_set\Environment1\Flying_path\Flying_circle\Position_UAV2.mat');
    Position_UAV3 = load('I:\Data_set\Environment1\Flying_path\Flying_circle\Position_UAV3.mat');
    Position_UAV4 = load('I:\Data_set\Environment1\Flying_path\Flying_circle\Position_UAV4.mat');
    Position_UAV5 = load('I:\Data_set\Environment1\Flying_path\Flying_circle\Position_UAV5.mat');
    Position_UAV6 = load('I:\Data_set\Environment1\Flying_path\Flying_circle\Position_UAV6.mat');
    Position_UAV7 = load('I:\Data_set\Environment1\Flying_path\Flying_circle\Position_UAV7.mat');
    Position_true(:,1) = Position_UAV1.Position_UAV1(:,i);
    Position_true(:,2) = Position_UAV2.Position_UAV2(:,i);
    Position_true(:,3) = Position_UAV3.Position_UAV3(:,i);
    Position_true(:,4) = Position_UAV4.Position_UAV4(:,i);
    Position_true(:,5) = Position_UAV5.Position_UAV5(:,i);
    Position_true(:,6) = Position_UAV6.Position_UAV6(:,i);
    Position_true(:,7) = Position_UAV7.Position_UAV7(:,i);
else
    Position_UAV1 = load('I:\Data_set\Environment1\Flying_path\Flying_disperse\Position_UAV1.mat');
    Position_UAV2 = load('I:\Data_set\Environment1\Flying_path\Flying_disperse\Position_UAV2.mat');
    Position_UAV3 = load('I:\Data_set\Environment1\Flying_path\Flying_disperse\Position_UAV3.mat');
    Position_UAV4 = load('I:\Data_set\Environment1\Flying_path\Flying_disperse\Position_UAV4.mat');
    Position_UAV5 = load('I:\Data_set\Environment1\Flying_path\Flying_disperse\Position_UAV5.mat');
    Position_UAV6 = load('I:\Data_set\Environment1\Flying_path\Flying_disperse\Position_UAV6.mat');
    Position_UAV7 = load('I:\Data_set\Environment1\Flying_path\Flying_disperse\Position_UAV7.mat');
    Position_true(:,1) = Position_UAV1.Position_UAV1(:,i);
    Position_true(:,2) = Position_UAV2.Position_UAV2(:,i);
    Position_true(:,3) = Position_UAV3.Position_UAV3(:,i);
    Position_true(:,4) = Position_UAV4.Position_UAV4(:,i);
    Position_true(:,5) = Position_UAV5.Position_UAV5(:,i);
    Position_true(:,6) = Position_UAV6.Position_UAV6(:,i);
    Position_true(:,7) = Position_UAV7.Position_UAV7(:,i);
    
end

position_true = Position_true' .* 100;

end